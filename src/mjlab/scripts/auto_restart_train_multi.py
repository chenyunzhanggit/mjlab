#!/usr/bin/env python3
"""Auto-restart script for training that monitors the process and restarts on failure.

This script:
1. Monitors a training process
2. Detects when it crashes (e.g., due to NaN/std issues)
3. Automatically finds the latest checkpoint
4. Restarts training with the latest checkpoint loaded

Usage:
    python -m mjlab.scripts.auto_restart_train <task_id> [training_args...]

Example:
    python -m mjlab.scripts.auto_restart_train Mjlab-Tracking-Teleoperation-Unitree-G1 \\
        --env.commands.motion.motion_path /path/to/motion \\
        --env.scene.num-envs 8192 \\
        --agent.max-iterations 300000 \\
        --gpu-ids 0 --gpu-ids 1

The script will automatically:
- Monitor the training process
- Restart on failure with latest checkpoint
- Continue until max_iterations is reached or manually stopped
"""

import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
# Script is in src/mjlab/scripts/
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
TRAIN_SCRIPT = project_root / "mjlab" / "scripts" / "train_multi.py"
sys.path.insert(0, str(project_root / "src"))


class TrainingMonitor:
  """Monitors training process and handles auto-restart."""

  def __init__(
    self,
    task_id: str,
    training_args: list[str],
    log_root: Path,
    experiment_name: str,
    max_restarts: int = 100,
    restart_delay: float = 5.0,
  ):
    self.task_id = task_id
    self.training_args = training_args
    self.log_root = log_root
    self.experiment_name = experiment_name
    self.max_restarts = max_restarts
    self.restart_delay = restart_delay
    self.restart_count = 0
    self.process: Optional[subprocess.Popen] = None
    self.original_run_name: Optional[str] = None
    self.is_first_run = True  # Track if this is the first run (not a restart)

  def find_latest_checkpoint(self) -> Optional[tuple[Path, str]]:
    """Find the latest checkpoint and its run directory.

    Returns:
        Tuple of (checkpoint_path, run_dir_name) or None if no checkpoint found
    """
    experiment_dir = self.log_root / "rsl_rl" / self.experiment_name
    if not experiment_dir.exists():
      print(f"[MONITOR] No experiment directory found: {experiment_dir}")
      return None

    # Find all run directories
    run_dirs = [
      d
      for d in experiment_dir.iterdir()
      if d.is_dir() and d.name != "wandb_checkpoints"
    ]

    if not run_dirs:
      print(f"[MONITOR] No run directories found in {experiment_dir}")
      return None

    # Sort by modification time (most recent first)
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Try to find checkpoint in each run directory (most recent first)
    for run_dir in run_dirs:
      checkpoints = list(run_dir.glob("model_*.pt"))
      if checkpoints:
        # Sort checkpoints by iteration number
        def get_iteration(ckpt: Path) -> int:
          match = re.search(r"model_(\d+)\.pt", ckpt.name)
          return int(match.group(1)) if match else 0

        checkpoints.sort(key=get_iteration)
        latest_checkpoint = checkpoints[-1]
        print(f"[MONITOR] Found latest checkpoint: {latest_checkpoint}")
        print(f"[MONITOR]   Run directory: {run_dir.name}")
        print(f"[MONITOR]   Iteration: {get_iteration(latest_checkpoint)}")
        return latest_checkpoint, run_dir.name

    print("[MONITOR] No checkpoints found in any run directory")
    return None

  def extract_run_name_from_args(self) -> Optional[str]:
    """Extract run_name from training arguments if present."""
    for i, arg in enumerate(self.training_args):
      if arg == "--agent.run-name" and i + 1 < len(self.training_args):
        return self.training_args[i + 1]
      elif arg.startswith("--agent.run-name="):
        return arg.split("=", 1)[1]
    return None

  def build_restart_command(
    self, checkpoint_info: Optional[tuple[Path, str]]
  ) -> list[str]:
    """Build the command to restart training with checkpoint."""
    cmd = [
      sys.executable,
      str(TRAIN_SCRIPT),
      self.task_id,
    ]

    # Add original training arguments
    cmd.extend(self.training_args)

    # If we have a checkpoint, add resume arguments
    if checkpoint_info is not None:
      checkpoint_path, run_dir_name = checkpoint_info

      # Remove any existing resume/load arguments to avoid conflicts
      filtered_args = []
      skip_next = False
      for i, arg in enumerate(cmd):
        if skip_next:
          skip_next = False
          continue
        if arg in ["--agent.resume", "--agent.load-run", "--agent.load-checkpoint"]:
          skip_next = True
          continue
        if (
          arg.startswith("--agent.resume=")
          or arg.startswith("--agent.load-run=")
          or arg.startswith("--agent.load-checkpoint=")
        ):
          continue
        filtered_args.append(arg)

      cmd = filtered_args

      # Add resume arguments
      cmd.extend(
        [
          "--agent.resume",
          "True",
          "--agent.load-run",
          run_dir_name,
          # "--agent.load-checkpoint", ".*",  # Use regex to get latest
        ]
      )

      print(f"[MONITOR] Restarting with checkpoint from run: {run_dir_name}")
    else:
      print("[MONITOR] Starting fresh training (no checkpoint found)")

    return cmd

  def run_training(self) -> int:
    """Run training and return exit code."""
    # Only look for checkpoint if this is a restart (not first run)
    checkpoint_info = None
    if not self.is_first_run:
      checkpoint_info = self.find_latest_checkpoint()
    else:
      print("[MONITOR] First run: starting fresh training (no checkpoint resume)")

    cmd = self.build_restart_command(checkpoint_info)

    print(f"[MONITOR] Resume checkpoint_info: {checkpoint_info}")
    print("[MONITOR] Starting training process...")
    print(f"[MONITOR] Command: {' '.join(cmd)}")
    print(f"[MONITOR] Restart count: {self.restart_count}/{self.max_restarts}")
    print("-" * 80)

    # Run the training process
    self.process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
    )

    # Stream output in real-time
    if self.process.stdout:
      for line in iter(self.process.stdout.readline, ""):
        if line:
          print(line.rstrip())
          sys.stdout.flush()

    # Wait for process to complete
    exit_code = self.process.wait()
    self.process = None

    return exit_code

  def is_successful_exit(self, exit_code: int) -> bool:
    """Check if exit code indicates successful completion."""
    # Exit code 0 means normal completion
    # Non-zero means error/crash
    return exit_code == 0

  def should_restart(self, exit_code: int) -> bool:
    """Determine if we should restart after this exit."""
    if self.is_successful_exit(exit_code):
      print(f"[MONITOR] Training completed successfully (exit code: {exit_code})")
      return False

    if self.restart_count >= self.max_restarts:
      print(f"[MONITOR] Maximum restart limit reached ({self.max_restarts})")
      return False

    print(f"[MONITOR] Training crashed (exit code: {exit_code})")
    return True

  def handle_signal(self, signum, frame):
    """Handle interrupt signals (Ctrl+C)."""
    print(f"\n[MONITOR] Received signal {signum}, stopping training process...")
    if self.process:
      print("[MONITOR] Terminating training process...")
      self.process.terminate()
      try:
        self.process.wait(timeout=10)
      except subprocess.TimeoutExpired:
        print("[MONITOR] Force killing training process...")
        self.process.kill()
        self.process.wait()
    sys.exit(0)

  def monitor(self):
    """Main monitoring loop."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, self.handle_signal)
    signal.signal(signal.SIGTERM, self.handle_signal)

    print("=" * 80)
    print("[MONITOR] Auto-restart training monitor started")
    print(f"[MONITOR] Task: {self.task_id}")
    print(f"[MONITOR] Experiment: {self.experiment_name}")
    print(f"[MONITOR] Max restarts: {self.max_restarts}")
    print("=" * 80)

    while True:
      exit_code = self.run_training()

      if not self.should_restart(exit_code):
        break

      self.restart_count += 1
      self.is_first_run = False  # Mark that this is now a restart
      print(f"\n[MONITOR] Waiting {self.restart_delay} seconds before restart...")
      time.sleep(self.restart_delay)
      print(f"[MONITOR] Restarting training (attempt {self.restart_count})...\n")

    print("=" * 80)
    print("[MONITOR] Monitoring stopped")
    print(f"[MONITOR] Total restarts: {self.restart_count}")
    print("=" * 80)


def main():
  """Main entry point."""
  if len(sys.argv) < 2:
    print(
      "Usage: python -m mjlab.scripts.auto_restart_train <task_id> [training_args...]"
    )
    print("\nExample:")
    print(
      "  python -m mjlab.scripts.auto_restart_train Mjlab-Tracking-Teleoperation-Unitree-G1 \\"
    )
    print("      --env.commands.motion.motion_path /path/to/motion \\")
    print("      --env.scene.num-envs 8192 \\")
    print("      --agent.max-iterations 300000 \\")
    print("      --gpu-ids 0 --gpu-ids 1")
    sys.exit(1)

  task_id = sys.argv[1]
  training_args = sys.argv[2:]

  # Extract experiment name from args or use default
  experiment_name = "g1_teleoperation"  # Default, can be overridden by args
  for i, arg in enumerate(training_args):
    if arg == "--agent.experiment-name" and i + 1 < len(training_args):
      experiment_name = training_args[i + 1]
      break
    elif arg.startswith("--agent.experiment-name="):
      experiment_name = arg.split("=", 1)[1]
      break

  log_root = Path("logs").resolve()

  monitor = TrainingMonitor(
    task_id=task_id,
    training_args=training_args,
    log_root=log_root,
    experiment_name=experiment_name,
    max_restarts=500,  # Adjust as needed
    restart_delay=2.5,  # Wait 5 seconds before restart
  )

  monitor.monitor()


if __name__ == "__main__":
  main()
