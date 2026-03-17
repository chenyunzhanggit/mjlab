"""PPO-AMP algorithm: PPO with Adversarial Motion Prior discriminator."""

from __future__ import annotations

from collections import (
  deque as _Deque,  # noqa: F401 – keep deque in scope for type hints
)

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from mjlab.tasks.tracking.rl.amp_discriminator import AMPDiscriminator, LossType
from mjlab.tasks.tracking.rl.circular_buffer import CircularBuffer


class PPOAmp(PPO):
  """PPO extended with an AMP discriminator.

  The discriminator is trained jointly with the policy inside ``update()``.
  Demo observations are sampled from ``demo_dataloader`` (a
  ``MultiMotionLoader`` already on GPU) — no extra data loading.

  Key differences from vanilla PPO:
    - ``process_env_step``: buffers policy AMP obs + sampled demo obs,
      mixes style reward with task reward before storage.
    - ``update``: interleaves PPO mini-batch update with discriminator
      gradient steps (same loop, matching the reference implementation).

  Discriminator obs shape convention: (B, disc_obs_steps, disc_obs_dim).
  When ``disc_obs_steps == 1`` this degenerates to single-frame AMP.
  """

  amp_discriminator: AMPDiscriminator

  def __init__(
    self,
    policy,
    storage,
    disc_obs_buffer: CircularBuffer,
    disc_demo_obs_buffer: CircularBuffer,
    *,
    amp_cfg: dict,
    **ppo_kwargs,
  ) -> None:
    # PPO.__init__ sets self.storage = None; we override it below.
    super().__init__(policy, **ppo_kwargs)
    self.storage = storage

    self.amp_cfg = amp_cfg
    disc_obs_dim: int = amp_cfg["disc_obs_dim"]
    disc_obs_steps: int = amp_cfg["disc_obs_steps"]
    loss_type_str: str = amp_cfg.get("loss_type", "LSGAN")
    loss_type = LossType[loss_type_str]

    self.amp_discriminator = AMPDiscriminator(
      disc_obs_dim=disc_obs_dim,
      disc_obs_steps=disc_obs_steps,
      hidden_dims=amp_cfg.get("disc_hidden_dims", (512, 256, 256)),
      activation=amp_cfg.get("disc_activation", "relu"),
      style_reward_scale=amp_cfg.get("style_reward_scale", 1.0),
      task_style_lerp=amp_cfg.get("task_style_lerp", 0.5),
      loss_type=loss_type,
      device=self.device,
    ).to(self.device)

    self.disc_optimizer = optim.Adam(
      [
        {
          "name": "disc_trunk",
          "params": self.amp_discriminator.disc_trunk.parameters(),
          "weight_decay": amp_cfg.get("disc_trunk_weight_decay", 1e-4),
        },
        {
          "name": "disc_linear",
          "params": self.amp_discriminator.disc_linear.parameters(),
          "weight_decay": amp_cfg.get("disc_linear_weight_decay", 0.0),
        },
      ],
      lr=amp_cfg.get("disc_learning_rate", 1e-4),
    )
    self.disc_max_grad_norm: float = amp_cfg.get("disc_max_grad_norm", 0.5)

    self.disc_obs_buffer = disc_obs_buffer
    self.disc_demo_obs_buffer = disc_demo_obs_buffer

    # Set externally by the runner before the first training step.
    self.demo_dataloader = None

    # Logging – per-step accumulators (kept for loss_dict)
    self.style_rewards: torch.Tensor | None = None
    self.rewards_lerp: torch.Tensor | None = None
    self._style_reward_sum: float = 0.0
    self._task_reward_sum: float = 0.0
    self._reward_steps: int = 0

    # Logging – episode-level accumulators (same scale as Mean reward)
    self.task_ep_rewbuffer: _Deque[float] = _Deque(maxlen=100)
    self.style_ep_rewbuffer: _Deque[float] = _Deque(maxlen=100)
    self._cur_task_ep_sum: torch.Tensor | None = None
    self._cur_style_ep_sum: torch.Tensor | None = None

  # ------------------------------------------------------------------
  # Override: buffer AMP obs, compute + mix style reward
  # ------------------------------------------------------------------

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict,
  ) -> None:
    # obs["amp"] shape: (num_envs, disc_obs_steps, disc_obs_dim)
    disc_obs = obs["amp"].to(self.device)
    self.disc_obs_buffer.append(disc_obs)
    # Sample a batch of demo sequences from the reference motion loader.
    if self.demo_dataloader is not None:
      num_envs = disc_obs.shape[0]
      demo_obs = self.demo_dataloader.sample_amp_obs_sequence(
        num_envs, self.amp_discriminator.disc_obs_steps
      )
      self.disc_demo_obs_buffer.append(demo_obs)

    # Style reward + lerp with task reward.
    self.style_rewards, _ = self.amp_discriminator.predict_style_reward(
      disc_obs, dt=self.amp_cfg.get("step_dt", 0.02)
    )
    self.rewards_lerp = self.amp_discriminator.lerp_reward(rewards, self.style_rewards)

    self._style_reward_sum += self.style_rewards.mean().item()
    self._task_reward_sum += rewards.mean().item()
    self._reward_steps += 1

    # Episode-level accumulation (same scale as Mean reward in the runner).
    if self._cur_task_ep_sum is None:
      self._cur_task_ep_sum = torch.zeros(rewards.shape[0], device=self.device)
      self._cur_style_ep_sum = torch.zeros(rewards.shape[0], device=self.device)
    self._cur_task_ep_sum += rewards
    self._cur_style_ep_sum += self.style_rewards
    new_ids = (dones > 0).nonzero(as_tuple=False)
    if new_ids.numel() > 0:
      self.task_ep_rewbuffer.extend(
        self._cur_task_ep_sum[new_ids][:, 0].cpu().numpy().tolist()
      )
      self.style_ep_rewbuffer.extend(
        self._cur_style_ep_sum[new_ids][:, 0].cpu().numpy().tolist()
      )
      self._cur_task_ep_sum[new_ids] = 0
      self._cur_style_ep_sum[new_ids] = 0

    super().process_env_step(obs, self.rewards_lerp, dones, extras)

  # ------------------------------------------------------------------
  # Override: PPO + discriminator update in one mini-batch loop
  # ------------------------------------------------------------------

  def update(self) -> dict:  # noqa: C901
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_disc_loss = 0.0
    mean_disc_grad_penalty = 0.0
    mean_disc_score = 0.0
    mean_disc_demo_score = 0.0

    if self.policy.is_recurrent:
      ppo_gen = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      ppo_gen = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    disc_gen = self.disc_obs_buffer.mini_batch_generator(
      fetch_length=self.storage.num_transitions_per_env,
      num_mini_batches=self.num_mini_batches,
      num_epochs=self.num_learning_epochs,
    )
    disc_demo_gen = self.disc_demo_obs_buffer.mini_batch_generator(
      fetch_length=self.storage.num_transitions_per_env,
      num_mini_batches=self.num_mini_batches,
      num_epochs=self.num_learning_epochs,
    )

    for samples, disc_obs_batch, disc_demo_obs_batch in zip(
      ppo_gen, disc_gen, disc_demo_gen, strict=False
    ):
      (
        obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        hid_states_batch,
        masks_batch,
      ) = samples

      original_batch_size = obs_batch.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
          )

      # ── Policy forward ────────────────────────────────────────
      self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
      actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
      value_batch = self.policy.evaluate(
        obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
      )
      mu_batch = self.policy.action_mean[:original_batch_size]
      sigma_batch = self.policy.action_std[:original_batch_size]
      entropy_batch = self.policy.entropy[:original_batch_size]

      # ── KL-adaptive learning rate ─────────────────────────────
      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            dim=-1,
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_t = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_t, src=0)
            self.learning_rate = lr_t.item()
          for pg in self.optimizer.param_groups:
            pg["lr"] = self.learning_rate

      # ── PPO loss ──────────────────────────────────────────────
      ratio = torch.exp(
        actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
      )
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        v_clip = target_values_batch + (value_batch - target_values_batch).clamp(
          -self.clip_param, self.clip_param
        )
        value_loss = torch.max(
          (value_batch - returns_batch).pow(2),
          (v_clip - returns_batch).pow(2),
        ).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      ppo_loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_batch.mean()
      )

      # ── Discriminator loss ────────────────────────────────────
      # disc_obs_batch / disc_demo_obs_batch: (mb, disc_obs_steps, disc_obs_dim)
      with torch.no_grad():
        disc_obs_normed = self.amp_discriminator.normalize_disc_obs(disc_obs_batch)
        disc_demo_normed = self.amp_discriminator.normalize_disc_obs(
          disc_demo_obs_batch
        )

      mb = disc_obs_normed.shape[0]
      disc_score = self.amp_discriminator(disc_obs_normed.reshape(mb, -1))
      disc_demo_score = self.amp_discriminator(disc_demo_normed.reshape(mb, -1))

      loss_type = self.amp_discriminator.loss_type
      if loss_type == LossType.GAN:
        bce = nn.BCEWithLogitsLoss()
        disc_loss = 0.5 * (
          bce(disc_score, torch.zeros_like(disc_score))
          + bce(disc_demo_score, torch.ones_like(disc_demo_score))
        )
      elif loss_type == LossType.LSGAN:
        disc_loss = 0.5 * (
          nn.MSELoss()(disc_score, -torch.ones_like(disc_score))
          + nn.MSELoss()(disc_demo_score, torch.ones_like(disc_demo_score))
        )
      elif loss_type == LossType.WGAN:
        disc_loss = -disc_demo_score.mean() + disc_score.mean()
      else:
        raise ValueError(f"Unknown LossType: {loss_type}")

      grad_penalty = self.amp_discriminator.compute_grad_penalty(
        disc_demo_normed.reshape(mb, -1),
        scale=self.amp_cfg.get("grad_penalty_scale", 5.0),
      )
      disc_total_loss = disc_loss + grad_penalty

      # ── Backward ─────────────────────────────────────────────
      self.optimizer.zero_grad()
      ppo_loss.backward()

      self.disc_optimizer.zero_grad()
      disc_total_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
      self.optimizer.step()

      nn.utils.clip_grad_norm_(
        self.amp_discriminator.parameters(), self.disc_max_grad_norm
      )
      self.disc_optimizer.step()

      # Update normaliser with un-normalised policy obs.
      self.amp_discriminator.update_normalization(disc_obs_batch)

      # ── Accumulate metrics ────────────────────────────────────
      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_batch.mean().item()
      mean_disc_loss += disc_loss.item()
      mean_disc_grad_penalty += grad_penalty.item()
      mean_disc_score += disc_score.mean().item()
      mean_disc_demo_score += disc_demo_score.mean().item()

    self.storage.clear()
    self._style_reward_sum = 0.0
    self._task_reward_sum = 0.0
    self._reward_steps = 0

    n = self.num_learning_epochs * self.num_mini_batches
    return {
      "value": mean_value_loss / n,
      "surrogate": mean_surrogate_loss / n,
      "entropy": mean_entropy / n,
      "amp/disc_loss": mean_disc_loss / n,
      "amp/disc_grad_penalty": mean_disc_grad_penalty / n,
      "amp/disc_score": mean_disc_score / n,
      "amp/disc_demo_score": mean_disc_demo_score / n,
    }
