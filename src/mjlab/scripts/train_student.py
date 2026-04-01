"""Training script for Phase-2 student knowledge-distillation.

Usage example
-------------
mjpython src/mjlab/scripts/train_student.py \\
    Mjlab-Tracking-Student-Unitree-G1 \\
    --env.commands.motion.motion-path /path/to/motions \\
    --env.scene.num-envs 4096 \\
    --agent.teacher-checkpoint /path/to/logs/rsl_rl/g1_teacher/run/model_50000.pt \\
    --agent.max-iterations 50000 \\
    --agent.distill-coef 1.0 \\
    --agent.distill-coef-decay 0.9999
"""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro

import mjlab.tasks  # noqa: F401 – populates the task registry
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.config.g1.rl_cfg import DistillationPpoRunnerCfg
from mjlab.tasks.tracking.mdp.multi_commands import MultiMotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags

_STUDENT_TASK = "Mjlab-Tracking-Student-Unitree-G1"


@dataclass(frozen=True)
class TrainStudentConfig:
  env: ManagerBasedRlEnvCfg
  agent: DistillationPpoRunnerCfg
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str = _STUDENT_TASK) -> "TrainStudentConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, DistillationPpoRunnerCfg)
    return TrainStudentConfig(env=env_cfg, agent=agent_cfg)


def _prepare_motion_for_student(cfg: TrainStudentConfig) -> None:
  """Prepare motion_files for multi-tracking student tasks.

  For consistency with ``train_multi.py``, we resolve motion sources *before*
  creating the environment:
    - Prefer ``env.commands.motion.motion_path`` if provided and exists.
    - Otherwise, use ``env.commands.motion.motion_files`` directly if non-empty.
    - Student script does not support WandB registry download; that is kept in ``train_multi.py``.
  """
  # Detect whether this is a multi tracking task with a motion command.
  if "motion" not in cfg.env.commands:
    return

  motion_cmd_cfg = cfg.env.commands["motion"]
  if not isinstance(motion_cmd_cfg, MultiMotionCommandCfg):
    return

  # Determine rank/world_size (single GPU treated as rank 0, world_size=1).
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    rank = 0
    world_size = 1
  else:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

  # Priority 1: motion_path
  motion_path = motion_cmd_cfg.motion_path
  if motion_path and motion_path.strip():
    motion_path = cast(str, motion_path)
    if not Path(motion_path).exists():
      raise ValueError(
        f"env.commands.motion.motion_path does not exist:\n  {motion_path}"
      )

    # Reuse the helper from train_multi for consistent behaviour.
    from mjlab.scripts.train_multi import get_data10K_motion_files, shard_motion_files

    print(
      f"[RANK {rank}/{world_size - 1}] Using local motion path for student distillation: {motion_path}",
      flush=True,
    )
    all_motion_files = get_data10K_motion_files(motion_path)
    motion_cmd_cfg.motion_files = shard_motion_files(all_motion_files, rank, world_size)
    return

  # Priority 2: motion_files already set (e.g., via config or CLI)
  if motion_cmd_cfg.motion_files and len(motion_cmd_cfg.motion_files) > 0:
    # For student script we simply trust this list; no extra sharding logic here
    # (multi-GPU student training can still work if motion_files was pre-sharded).
    print(
      f"[RANK {rank}/{world_size - 1}] Student motion_files already set ({len(motion_cmd_cfg.motion_files)} files).",
      flush=True,
    )
    return

  # Otherwise, configuration is incomplete for tracking student task.
  raise ValueError(
    "For tracking student tasks, provide at least one of:\n"
    "  --env.commands.motion.motion-path /path/to/motion/directory\n"
    "  --env.commands.motion.motion-files /path/to/file1.npz /path/to/file2.npz ..."
  )


def run_train(task_id: str, cfg: TrainStudentConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  # Prepare motion files for multi-tracking student tasks (if applicable)
  _prepare_motion_for_student(cfg)
  cfg.agent.seed = seed
  cfg.env.seed = seed

  if not cfg.agent.teacher_checkpoint:
    raise ValueError(
      "teacher_checkpoint must be set.  "
      "Pass --agent.teacher-checkpoint /path/to/model_N.pt"
    )
  if not Path(cfg.agent.teacher_checkpoint).exists():
    raise FileNotFoundError(
      f"Teacher checkpoint not found: {cfg.agent.teacher_checkpoint}"
    )

  print(f"[INFO] Distillation training: device={device}, seed={seed}, rank={rank}")
  print(f"[INFO] Teacher checkpoint: {cfg.agent.teacher_checkpoint}")

  env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
  log_root_path = log_dir.parent

  resume_path: Path | None = None
  if cfg.agent.resume:
    resume_path = get_checkpoint_path(
      log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
    )

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  assert runner_cls is not None
  runner = runner_cls(env, agent_cfg, str(log_dir), device)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)

  if resume_path is not None:
    print(f"[INFO] Resuming from checkpoint: {resume_path}")
    runner.load(str(resume_path))

  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )
  env.close()


def launch_training(task_id: str, args: TrainStudentConfig | None = None) -> None:
  args = args or TrainStudentConfig.from_task(task_id)  # type: ignore

  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    run_train(task_id, args, log_dir)
  else:
    import torchrunx

    logging.basicConfig(level=logging.INFO)
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching distillation training with {num_gpus} GPUs")
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main() -> None:
  all_tasks = list_tasks()
  # Default to the student task; allow override for future multi-robot variants.
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    args=[_STUDENT_TASK] + sys.argv[1:],
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,  # type: ignore[attr-defined]
  )

  args = tyro.cli(
    TrainStudentConfig,
    args=remaining_args,
    default=TrainStudentConfig.from_task(chosen_task),  # type: ignore
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,  # type: ignore[attr-defined]
  )

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
