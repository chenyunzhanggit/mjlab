"""Knowledge-distillation runner for motion-tracking tasks.

Phase-2 training pipeline:
  1. Rollout with **student** policy (sees reduced 'student' obs group).
  2. In parallel, record frozen **teacher** action means from the 'policy'
     (teacher) obs group.
  3. Distillation update: minimise MSE between student and teacher action
     means over the collected rollout buffer.
"""

from __future__ import annotations

import os
import time
from collections import deque

import torch
import torch.nn.functional as F
import wandb
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.modules import ActorCritic

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl.exporter import attach_onnx_metadata, export_policy_as_onnx
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner


class MotionTrackingDistillationRunner(MotionTrackingOnPolicyRunner):
  """Pure knowledge-distillation runner for student policy training.

  The frozen teacher is loaded from ``train_cfg['teacher_checkpoint']``
  and is expected to have been trained with the standard
  ``'Mjlab-Tracking-Teleoperation-Unitree-G1'`` task (i.e. its actor
  reads the ``'policy'`` observation group which contains privileged
  future reference-motion frames).

  The student actor reads the much smaller ``'student'`` observation group
  and is trained purely via MSE distillation loss against teacher action means.
  No PPO update is performed.
  """

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    # Build student ActorCritic + PPO algorithm via the base class.
    super().__init__(env, train_cfg, log_dir, device, registry_name)

    # ── Distillation hyper-parameters ──────────────────────────────────
    self.distill_coef: float = float(train_cfg.get("distill_coef", 1.0))
    self.distill_coef_decay: float = float(train_cfg.get("distill_coef_decay", 0.9999))
    self.distill_epochs: int = int(train_cfg.get("distill_epochs", 2))
    self.distill_num_mini_batches: int = int(
      train_cfg.get("distill_num_mini_batches", 4)
    )

    # ── Build and load frozen teacher ──────────────────────────────────
    teacher_checkpoint: str = train_cfg.get("teacher_checkpoint", "")
    teacher_actor_hidden_dims: tuple[int, ...] = tuple(
      train_cfg.get("teacher_actor_hidden_dims", (2048, 2048, 1024, 1024, 512, 512))
    )

    if teacher_checkpoint:
      self.teacher = self._build_teacher(teacher_checkpoint, teacher_actor_hidden_dims)
    else:
      self.teacher = None
      print("[Distill] No teacher_checkpoint — running in inference/play mode.")

    # ── Pre-compute student obs dimension from first env observation ───
    # obs["student"] is already available after base __init__ calls
    # env.reset() inside RslRlVecEnvWrapper.
    init_obs = self.env.get_observations().to(self.device)
    self._student_obs_dim: int = init_obs["student"].shape[-1]
    self._teacher_obs_dim: int = init_obs["policy"].shape[-1]

  # ──────────────────────────────────────────────────────────────────────
  # Public API
  # ──────────────────────────────────────────────────────────────────────

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """Pure distillation training loop.

    Rollout 只用于收集 (student_obs, teacher_action) 配对，不做任何 PPO 计算。
    每轮迭代执行一次 MSE 蒸馏更新。
    """
    self._prepare_logging_writer()

    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.train_mode()

    ep_infos: list = []
    rewbuffer: deque = deque(maxlen=100)
    lenbuffer: deque = deque(maxlen=100)
    cur_reward_sum = torch.zeros(
      self.env.num_envs, dtype=torch.float, device=self.device
    )
    cur_episode_length = torch.zeros(
      self.env.num_envs, dtype=torch.float, device=self.device
    )

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    num_steps = self.cfg["num_steps_per_env"]
    num_envs = self.env.num_envs
    num_actions = self.env.num_actions
    teacher_actions_buf = torch.zeros(
      num_steps, num_envs, num_actions, device=self.device
    )
    student_obs_buf = torch.zeros(
      num_steps, num_envs, self._student_obs_dim, device=self.device
    )

    start_iter = self.current_learning_iteration
    tot_iter = start_iter + num_learning_iterations
    for it in range(start_iter, tot_iter):
      start = time.time()

      # ── Rollout: collect (student_obs, teacher_action) pairs ──────────
      with torch.inference_mode():
        for step in range(self.num_steps_per_env):
          # Student forward（纯推理，不走 PPO buffer）
          s_obs = obs["student"]
          s_obs_norm = (
            self.alg.policy.actor_obs_normalizer(s_obs)
            if self.alg.policy.actor_obs_normalization
            else s_obs
          )
          actions = self.alg.policy.actor(s_obs_norm)

          teacher_actions_buf[step].copy_(self._teacher_action_mean(obs["policy"]))
          student_obs_buf[step].copy_(s_obs)

          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))  # type: ignore[union-attr]
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )

          if self.log_dir is not None:
            if "episode" in extras:
              ep_infos.append(extras["episode"])
            elif "log" in extras:
              ep_infos.append(extras["log"])

            cur_reward_sum += rewards
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

      collection_time = time.time() - start
      start = time.time()

      # ── Distillation update ───────────────────────────────────────────
      distill_loss_val = self._distillation_update(
        student_obs_buf.view(-1, self._student_obs_dim),
        teacher_actions_buf.view(-1, num_actions),
      )

      learn_time = time.time() - start
      self.current_learning_iteration = it

      if self.log_dir is not None and not self.disable_logs:
        self.log(
          {
            "it": it,
            "start_iter": start_iter,
            "tot_iter": tot_iter,
            "num_learning_iterations": num_learning_iterations,
            "collection_time": collection_time,
            "learn_time": learn_time,
            "loss_dict": {"distill_loss": distill_loss_val},
            "rewbuffer": rewbuffer,
            "lenbuffer": lenbuffer,
            "ep_infos": ep_infos,
          }
        )
        if it % self.save_interval == 0:
          self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

      ep_infos.clear()

    if self.log_dir is not None and not self.disable_logs:
      self.save(
        os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
      )

  def save(self, path: str, infos=None):
    """Save student checkpoint + ONNX export."""
    # Persist environment state (common_step_counter) in infos.
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}

    # Delegate to rsl_rl OnPolicyRunner.save (skips the middle layer).
    from rsl_rl.runners import OnPolicyRunner

    OnPolicyRunner.save(self, path, infos)

    # Export student policy as ONNX.
    policy_path = path.split("model")[0]
    filename = policy_path.split("/")[-2] + ".onnx"
    normalizer = (
      self.alg.policy.actor_obs_normalizer
      if self.alg.policy.actor_obs_normalization
      else None
    )
    export_policy_as_onnx(
      self.env.unwrapped,
      self.alg.policy,
      normalizer=normalizer,
      path=policy_path,
      filename=filename,
    )
    run_name = wandb.run.name if self.logger_type == "wandb" and wandb.run else "local"
    attach_onnx_metadata(
      self.env.unwrapped,
      run_name,  # type: ignore[arg-type]
      path=policy_path,
      filename=filename,
    )
    if self.logger_type == "wandb":
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
      if self.registry_name is not None:
        wandb.run.use_artifact(self.registry_name)  # type: ignore[union-attr]
        self.registry_name = None

  def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
    """Override to avoid accessing policy.action_std before distribution is set.

    In distillation, we call policy.actor() directly without update_distribution(),
    so policy.distribution is None.  We read the std parameter directly instead.
    """
    policy = self.alg.policy
    # Monkey-patch a dummy distribution so the parent log() can read action_std
    # without crashing.  We restore it immediately after.
    if policy.distribution is None:
      import torch.distributions as D

      if hasattr(policy, "std"):
        _std = policy.std.detach()
      elif hasattr(policy, "log_std"):
        _std = policy.log_std.detach().exp()
      else:
        _std = torch.zeros(self.env.num_actions, device=self.device)
      _dummy_mean = torch.zeros_like(_std)
      policy.distribution = D.Normal(_dummy_mean, _std)
      super().log(locs, width=width, pad=pad)
      policy.distribution = None
    else:
      super().log(locs, width=width, pad=pad)

  # ──────────────────────────────────────────────────────────────────────
  # Private helpers
  # ──────────────────────────────────────────────────────────────────────

  def _build_teacher(
    self, checkpoint_path: str, actor_hidden_dims: tuple[int, ...]
  ) -> ActorCritic:
    """Instantiate teacher ActorCritic, load checkpoint, and freeze it."""
    # Get current observations to infer input dimensions.
    obs = self.env.get_observations().to(self.device)
    teacher_obs_groups = {"policy": ("policy",), "critic": ("critic",)}

    teacher = ActorCritic(
      obs,
      teacher_obs_groups,
      self.env.num_actions,
      actor_hidden_dims=actor_hidden_dims,
      critic_hidden_dims=actor_hidden_dims,
      activation="elu",
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      noise_std_type="scalar",
    ).to(self.device)

    ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    for param in teacher.parameters():
      param.requires_grad_(False)

    print(
      f"[Distill] Loaded frozen teacher from: {checkpoint_path}  "
      f"(iter {ckpt.get('iter', '?')})"
    )
    return teacher

  @torch.no_grad()
  def _teacher_action_mean(self, teacher_obs: torch.Tensor) -> torch.Tensor:
    """Forward the frozen teacher actor and return action means."""
    assert self.teacher is not None, "_teacher_action_mean called without a teacher"
    if self.teacher.actor_obs_normalization:
      obs_norm = self.teacher.actor_obs_normalizer(teacher_obs)
    else:
      obs_norm = teacher_obs
    return self.teacher.actor(obs_norm)

  def _distillation_update(
    self,
    student_obs_flat: torch.Tensor,
    teacher_actions_flat: torch.Tensor,
  ) -> float:
    """Run distillation gradient steps and return the mean loss value."""
    total_samples = student_obs_flat.shape[0]
    batch_size = max(1, total_samples // self.distill_num_mini_batches)
    total_loss = 0.0
    n_updates = 0

    policy = self.alg.policy
    optimizer = self.alg.optimizer
    max_grad_norm: float = self.alg_cfg.get("max_grad_norm", 1.0)

    policy.actor.train()

    for _ in range(self.distill_epochs):
      perm = torch.randperm(total_samples, device=self.device)
      for i in range(self.distill_num_mini_batches):
        idx = perm[i * batch_size : (i + 1) * batch_size]
        s_obs = student_obs_flat[idx]
        t_act = teacher_actions_flat[idx]

        # Normalise student obs if enabled.
        if policy.actor_obs_normalization:
          s_obs_norm = policy.actor_obs_normalizer(s_obs)
        else:
          s_obs_norm = s_obs

        s_mean = policy.actor(s_obs_norm)

        loss = F.mse_loss(s_mean, t_act)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
          policy.actor.parameters(), max_norm=max_grad_norm
        )
        optimizer.step()

        total_loss += loss.item()
        n_updates += 1

    return total_loss / max(n_updates, 1)
