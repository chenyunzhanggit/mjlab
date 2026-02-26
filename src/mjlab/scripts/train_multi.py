"""Script to train RL agent with RSL-RL."""

import glob
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, CustomManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.multi_commands import MultiMotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  registry_name: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def get_data10K_motion_files(motion_path: str, recursive: bool = True) -> list[str]:
    if os.path.isfile(motion_path):
        # 如果是单个文件，返回包含该文件的列表
        if not motion_path.lower().endswith('.npz'):
            print(f"警告: {motion_path} 不是.npz文件")
        return [motion_path]
    
    elif os.path.isdir(motion_path):
        if recursive:
            # 递归搜索所有子目录中的.npz文件
            pattern = os.path.join(motion_path, "**", "*.npz")
            motion_files = glob.glob(pattern, recursive=True)
        else:
            # 只搜索当前目录（原行为）
            motion_files = glob.glob(os.path.join(motion_path, "*.npz"))
        
        if not motion_files:
            raise ValueError(f"在目录中未找到.npz文件: {motion_path}")
        
        # 过滤掉目录（确保只返回文件）
        motion_files = [f for f in motion_files if os.path.isfile(f)]
        motion_files.sort()  # 排序确保一致性
        
        print(f"在 {motion_path} 中找到 {len(motion_files)} 个motion文件{'（包含子目录）' if recursive else ''}")
        # Only print first few files to avoid log spam (especially in multi-GPU training)
        max_print = 5
        for i, file in enumerate(motion_files[:max_print]):
            # 显示相对路径，更清晰
            rel_path = os.path.relpath(file, motion_path)
            print(f" - {rel_path}")
        if len(motion_files) > max_print:
            print(f" - ... 还有 {len(motion_files) - max_print} 个文件未显示")
        
        # Verify no duplicates
        unique_files = len(set(motion_files))
        if unique_files != len(motion_files):
            print(f"警告: 发现 {len(motion_files) - unique_files} 个重复文件！")
        else:
            print(f"验证: 所有 {len(motion_files)} 个文件都是唯一的")
        
        return motion_files
    
    else:
        raise ValueError(f"无效路径: {motion_path}。必须是文件或目录。")


def shard_motion_files(motion_files: list[str], rank: int, world_size: int) -> list[str]:
    """Shard motion files across multiple GPUs to reduce memory usage.
    
    Each GPU will only load a subset of the motion files, reducing memory/VRAM usage.
    This is especially useful when training with large datasets.
    
    Args:
        motion_files: List of all motion file paths
        rank: Current process rank (0 to world_size-1)
        world_size: Total number of processes/GPUs
        
    Returns:
        List of motion files assigned to this rank
        
    Raises:
        ValueError: If world_size > num_files (not enough files for all GPUs)
    """
    if world_size <= 1:
        return motion_files
    
    num_files = len(motion_files)
    
    # Ensure we have enough files for all GPUs
    if num_files < world_size:
        raise ValueError(
            f"Not enough motion files ({num_files}) for {world_size} GPUs. "
            f"Each GPU needs at least 1 file. Consider using fewer GPUs or adding more motion files."
        )
    
    # Calculate shard size (round up to ensure all files are distributed)
    shard_size = (num_files + world_size - 1) // world_size  # Ceiling division
    
    # Calculate start and end indices for this rank
    start_idx = rank * shard_size
    end_idx = min(start_idx + shard_size, num_files)
    
    # Get files for this rank
    sharded_files = motion_files[start_idx:end_idx]
    
    # Log information about the sharding
    print(
        f"[RANK {rank}/{world_size-1}] Data sharding: "
        f"assigned {len(sharded_files)}/{num_files} motion files "
        f"(indices {start_idx}-{end_idx-1})"
    )
    if len(sharded_files) > 0:
        # Show full path or relative path to make it clear files are different
        first_file = sharded_files[0]
        # Try to show relative path if possible
        try:
            # Find common base directory
            common_base = os.path.commonpath(sharded_files) if len(sharded_files) > 1 else os.path.dirname(first_file)
            rel_first = os.path.relpath(first_file, common_base)
            print(f"[RANK {rank}] First file: {rel_first} (full: {first_file})")
        except (ValueError, OSError):
            # If paths are on different drives or can't compute common path, show full path
            print(f"[RANK {rank}] First file (full path): {first_file}")
        
        if len(sharded_files) > 1:
            last_file = sharded_files[-1]
            try:
                common_base = os.path.commonpath(sharded_files)
                rel_last = os.path.relpath(last_file, common_base)
                print(f"[RANK {rank}] Last file: {rel_last} (full: {last_file})")
            except (ValueError, OSError):
                print(f"[RANK {rank}] Last file (full path): {last_file}")
        
        # Verify files are actually different
        if len(sharded_files) > 1:
            unique_files = len(set(sharded_files))
            if unique_files != len(sharded_files):
                print(f"[RANK {rank}] WARNING: Found {len(sharded_files) - unique_files} duplicate files in shard!")
            else:
                print(f"[RANK {rank}] Verified: All {len(sharded_files)} files are unique")
    
    return sharded_files


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
    world_size = 1
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}, world_size={world_size}")

  registry_name: str | None = None

  # Check if this is a multi tracking task by checking for motion command.
  is_multi_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MultiMotionCommandCfg
  )

  if is_multi_tracking_task:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MultiMotionCommandCfg)
    # Priority 1: If motion_path is set and exists, get files from path and shard
    # (This takes priority to ensure we get the complete file list from the source)
    if motion_cmd.motion_path and motion_cmd.motion_path.strip() and Path(motion_cmd.motion_path).exists():
      print(f"[RANK {rank}] Using local motion path: {motion_cmd.motion_path}")
      # Get all motion files first from the path
      all_motion_files = get_data10K_motion_files(motion_cmd.motion_path)
      # Shard files across GPUs to reduce memory usage
      motion_cmd.motion_files = shard_motion_files(all_motion_files, rank, world_size)
    
    # Priority 2: If motion_files are already set (via CLI or config), shard them
    # (Only used when motion_path is not set)
    elif motion_cmd.motion_files and len(motion_cmd.motion_files) > 0:
      print(f"[RANK {rank}] Motion files already set, sharding {len(motion_cmd.motion_files)} files across {world_size} GPUs")
      motion_cmd.motion_files = shard_motion_files(motion_cmd.motion_files, rank, world_size)

    # Priority 3: Download from WandB registry
    elif cfg.registry_name:
      # Download from WandB registry.
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_path = str(Path(artifact.download()) / "motion.npz")
      # For WandB artifacts, if it's a directory, shard the files
      if os.path.isdir(motion_cmd.motion_path):
        all_motion_files = get_data10K_motion_files(motion_cmd.motion_path)
        motion_cmd.motion_files = shard_motion_files(all_motion_files, rank, world_size)
  

    else:
      raise ValueError(
        "For tracking tasks, provide either:\n"
        "  --registry-name your-org/motions/motion-name (download from WandB)\n"
        "  --env.commands.motion.motion-path /path/to/motion/directory (local directory)\n"
        "  --env.commands.motion.motion-files /path/to/file1.npz /path/to/file2.npz ... (list of files)"
      )

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

#   use_custom_env = True  # Set to False to use default ManagerBasedRlEnv
  
#   if use_custom_env:
#     env = CustomManagerBasedRlEnv(
#       cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
#     )
#   else:
  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )
  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  if is_multi_tracking_task:
    runner_kwargs["registry_name"] = registry_name
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
