"""Script to play a trained student policy (knowledge-distillation Phase-2).

Usage example
-------------
mjpython src/mjlab/scripts/play_student.py \
    --checkpoint-file logs/rsl_rl/g1_student/2024-01-01_00-00-00/model_50000.pt \
    --motion-path /path/to/motions \
    --num-envs 1

Or with WandB run path:
    --wandb-run-path entity/project/run_id \
    --motion-path /path/to/motions
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

import mjlab
import mjlab.tasks  # noqa: F401 — populates task registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.rl.exporter import export_policy_as_onnx
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

_STUDENT_TASK = "Mjlab-Tracking-Student-Unitree-G1"


@dataclass(frozen=True)
class PlayStudentConfig:
  checkpoint_file: str | None = None
  """Path to the student model checkpoint (.pt file)."""

  wandb_run_path: str | None = None
  """WandB run path (entity/project/run_id) to download the checkpoint from."""

  motion_path: str | None = None
  """Directory containing .npz motion files."""

  motion_files: list[str] | None = None
  """Explicit list of .npz motion file paths (alternative to motion_path)."""

  num_envs: int = 1
  device: str | None = None

  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None

  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False


def main() -> None:
  cfg = tyro.cli(PlayStudentConfig, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(_STUDENT_TASK, play=True)
  agent_cfg = load_rl_cfg(_STUDENT_TASK)

  # ── Motion files ──────────────────────────────────────────────────────
  from mjlab.tasks.tracking.mdp.multi_commands import MultiMotionCommandCfg

  if "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MultiMotionCommandCfg
  ):
    motion_cmd = env_cfg.commands["motion"]
    if cfg.motion_path and Path(cfg.motion_path).exists():
      import glob

      files = sorted(
        glob.glob(os.path.join(cfg.motion_path, "**", "*.npz"), recursive=True)
      )
      if not files:
        raise ValueError(f"No .npz files found in: {cfg.motion_path}")
      motion_cmd.motion_files = files
      print(f"[INFO] Found {len(files)} motion file(s) in {cfg.motion_path}")
    elif cfg.motion_files:
      motion_cmd.motion_files = cfg.motion_files
    else:
      raise ValueError(
        "Provide either --motion-path <dir> or --motion-files <file1> <file2> ..."
      )

  # ── Terminations ──────────────────────────────────────────────────────
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO] Terminations disabled")

  # ── Checkpoint ────────────────────────────────────────────────────────
  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
  elif cfg.wandb_run_path is not None:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    resume_path, was_cached = get_wandb_checkpoint_path(
      log_root_path, Path(cfg.wandb_run_path)
    )
    print(
      f"[INFO] Checkpoint: {resume_path.name} "
      f"({'cached' if was_cached else 'downloaded'})"
    )
  else:
    raise ValueError("Provide --checkpoint-file or --wandb-run-path.")

  log_dir = resume_path.parent

  # ── Environment ───────────────────────────────────────────────────────
  env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if cfg.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if cfg.video:
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # ── Load student runner & policy ──────────────────────────────────────
  # MotionTrackingDistillationRunner skips teacher loading when
  # teacher_checkpoint is empty (play mode).
  runner_cls = load_runner_cls(_STUDENT_TASK) or OnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  # ── Export ONNX ───────────────────────────────────────────────────────
  export_model_dir = str(log_dir / "exported")
  ckpt_num = resume_path.stem.split("_")[-1]
  export_policy_as_onnx(
    env=env.unwrapped,
    actor_critic=runner.alg.policy,
    path=export_model_dir,
    normalizer=runner.alg.policy.actor_obs_normalizer
    if runner.alg.policy.actor_obs_normalization
    else None,
    filename=f"student_{ckpt_num}.onnx",
    verbose=False,
  )

  # ── Viewer ────────────────────────────────────────────────────────────
  env.get_observations()
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer: {resolved_viewer}")

  env.close()


if __name__ == "__main__":
  main()
