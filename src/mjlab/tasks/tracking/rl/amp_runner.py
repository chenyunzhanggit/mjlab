"""AMP runner: wires up PPOAmp using the env's already-loaded motion data."""

from __future__ import annotations

import os
import statistics
import types

import torch

from mjlab.rl.runner import _compute_returns_global_norm
from mjlab.tasks.tracking.mdp.multi_commands import MultiMotionCommand
from mjlab.tasks.tracking.rl.circular_buffer import CircularBuffer
from mjlab.tasks.tracking.rl.ppo_amp import PPOAmp
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner


class AmpMotionTrackingOnPolicyRunner(MotionTrackingOnPolicyRunner):
  """Runner that replaces PPO with PPOAmp.

  Changes vs. the base runner:
    1. ``_construct_algorithm``: builds PPOAmp, creates CircularBuffers for
       policy and demo AMP observations, resolves disc obs dims from the env.
    2. ``save`` / ``load``: persist discriminator weights + normaliser.
    3. ``train_mode`` / ``eval_mode``: toggle discriminator train/eval state.
  """

  alg: PPOAmp

  def _construct_algorithm(self, obs):
    amp_cfg: dict = dict(self.cfg.get("amp", {}))

    # ------------------------------------------------------------------
    # Resolve disc obs dims from obs["amp"] (shape: num_envs, steps, dim)
    # ------------------------------------------------------------------
    if "amp" not in obs:
      raise KeyError(
        "Environment must provide an 'amp' observation group for AMP training."
      )
    amp_obs = obs["amp"]
    if amp_obs.dim() != 3:
      raise ValueError(
        f"obs['amp'] must be 3-D (num_envs, disc_obs_steps, disc_obs_dim), "
        f"got shape {tuple(amp_obs.shape)}."
      )
    disc_obs_steps: int = amp_obs.shape[1]
    disc_obs_dim: int = amp_obs.shape[2]
    amp_cfg["disc_obs_steps"] = disc_obs_steps
    amp_cfg["disc_obs_dim"] = disc_obs_dim

    # step_dt for reward scaling
    amp_cfg.setdefault("step_dt", float(self.env.unwrapped.step_dt))

    # ------------------------------------------------------------------
    # Build base PPO (creates actor_critic + storage via init_storage).
    # MjlabOnPolicyRunner applies the global-norm compute_returns patch.
    # ------------------------------------------------------------------
    ppo = super()._construct_algorithm(obs)

    # ------------------------------------------------------------------
    # Access the motion loader already on GPU inside MultiMotionCommand.
    # ------------------------------------------------------------------
    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    assert isinstance(motion_cmd, MultiMotionCommand)
    motion_loader = motion_cmd.motion  # MultiMotionLoader

    # AMP obs velocities must be expressed in the same frame as the env's IMU
    # sensor.  The IMU body name is stored in amp_cfg["imu_body_name"]; if not
    # set, fall back to anchor_body_name (legacy behaviour).
    imu_body_name: str = amp_cfg.get("imu_body_name") or motion_cmd.cfg.anchor_body_name
    anchor_body_idx = motion_loader.mujoco_body_names.index(imu_body_name)
    motion_loader.build_amp_obs_buffer(anchor_body_idx)
    motion_loader.build_amp_seq_table(disc_obs_steps)

    # ------------------------------------------------------------------
    # CircularBuffers for policy and demo AMP observations.
    # ------------------------------------------------------------------
    buf_size: int = amp_cfg.get("disc_obs_buffer_size", 100)
    disc_obs_buffer = CircularBuffer(
      max_len=buf_size,
      batch_size=self.env.num_envs,
      device=self.device,
    )
    disc_demo_obs_buffer = CircularBuffer(
      max_len=buf_size,
      batch_size=self.env.num_envs,
      device=self.device,
    )

    # ------------------------------------------------------------------
    # Build PPOAmp, reuse storage already initialised by the base PPO.
    # ------------------------------------------------------------------
    alg = PPOAmp(
      ppo.policy,
      ppo.storage,
      disc_obs_buffer,
      disc_demo_obs_buffer,
      amp_cfg=amp_cfg,
      device=self.device,
      num_learning_epochs=ppo.num_learning_epochs,
      num_mini_batches=ppo.num_mini_batches,
      clip_param=ppo.clip_param,
      gamma=ppo.gamma,
      lam=ppo.lam,
      value_loss_coef=ppo.value_loss_coef,
      entropy_coef=ppo.entropy_coef,
      learning_rate=ppo.learning_rate,
      max_grad_norm=ppo.max_grad_norm,
      use_clipped_value_loss=ppo.use_clipped_value_loss,
      desired_kl=ppo.desired_kl,
      schedule=ppo.schedule,
      normalize_advantage_per_mini_batch=ppo.normalize_advantage_per_mini_batch,
      multi_gpu_cfg=self.multi_gpu_cfg,
    )

    # Re-apply the global-norm advantage patch.
    alg.compute_returns = types.MethodType(_compute_returns_global_norm, alg)

    # Inject motion loader for demo observation sampling.
    alg.demo_dataloader = motion_loader  # type: ignore[assignment]

    return alg

  # ------------------------------------------------------------------
  # Persist discriminator alongside policy checkpoint
  # ------------------------------------------------------------------

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    disc_path = path.replace(".pt", "_disc.pt")
    torch.save(
      {
        "discriminator": self.alg.amp_discriminator.state_dict(),
        "disc_obs_normalizer": self.alg.amp_discriminator.disc_obs_normalizer.state_dict(),
        "disc_optimizer": self.alg.disc_optimizer.state_dict(),
      },
      disc_path,
    )

  def load(
    self, path: str, load_optimizer: bool = True, map_location: str | None = None
  ):
    infos = super().load(path, load_optimizer, map_location)
    disc_path = path.replace(".pt", "_disc.pt")
    if os.path.exists(disc_path):
      disc_dict = torch.load(disc_path, weights_only=False, map_location=map_location)
      self.alg.amp_discriminator.load_state_dict(disc_dict["discriminator"])
      self.alg.amp_discriminator.disc_obs_normalizer.load_state_dict(
        disc_dict["disc_obs_normalizer"]
      )
      if load_optimizer:
        self.alg.disc_optimizer.load_state_dict(disc_dict["disc_optimizer"])
    return infos

  def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
    task_buf = self.alg.task_ep_rewbuffer
    style_buf = self.alg.style_ep_rewbuffer
    if len(task_buf) > 0 and len(style_buf) > 0:
      locs["loss_dict"]["amp/task_reward"] = statistics.mean(task_buf)
      locs["loss_dict"]["amp/style_reward"] = statistics.mean(style_buf)
    super().log(locs, width, pad)

  def train_mode(self) -> None:
    super().train_mode()
    self.alg.amp_discriminator.train()
    self.alg.amp_discriminator.disc_obs_normalizer.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    self.alg.amp_discriminator.eval()
    self.alg.amp_discriminator.disc_obs_normalizer.eval()
