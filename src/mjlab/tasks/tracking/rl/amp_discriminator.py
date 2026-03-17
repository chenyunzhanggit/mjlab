"""AMP Discriminator module — adapted from atom01_train rsl_rl."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation
from torch import autograd


class LossType(Enum):
  GAN = 0
  LSGAN = 1
  WGAN = 2


class AMPDiscriminator(nn.Module):
  """MLP discriminator with per-frame observation normalisation and history support.

  Input layout per frame: [lin_vel_b (3), ang_vel_b (3), joint_pos (n_dof), joint_vel (n_dof)]
  The discriminator receives ``disc_obs_steps`` consecutive frames concatenated flat.
  """

  def __init__(
    self,
    disc_obs_dim: int,
    disc_obs_steps: int,
    hidden_dims: list[int] | tuple[int, ...] = (512, 256, 256),
    activation: str = "relu",
    style_reward_scale: float = 1.0,
    task_style_lerp: float = 0.5,
    loss_type: LossType = LossType.LSGAN,
    device: str = "cpu",
  ):
    super().__init__()

    self.disc_obs_dim = disc_obs_dim
    self.disc_obs_steps = disc_obs_steps
    self.input_dim = disc_obs_dim * disc_obs_steps

    assert style_reward_scale >= 0, "style_reward_scale must be non-negative."
    self.style_reward_scale = style_reward_scale
    self.task_style_lerp = task_style_lerp
    self.loss_type = loss_type
    self.device = device

    act = resolve_nn_activation(activation)

    # Per-frame observation normaliser (shared across steps).
    self.disc_obs_normalizer = EmpiricalNormalization(shape=disc_obs_dim, until=1e8).to(
      device
    )

    # Trunk
    layers: list[nn.Module] = []
    in_dim = self.input_dim
    for h in hidden_dims:
      layers += [nn.Linear(in_dim, h), act]
      in_dim = h
    self.disc_trunk = nn.Sequential(*layers)
    self.disc_linear = nn.Linear(in_dim, 1)

    if self.loss_type == LossType.WGAN:
      self.disc_output_normalizer = EmpiricalNormalization(shape=1, until=1e8).to(
        device
      )
    else:
      self.disc_output_normalizer = nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """x: (batch, disc_obs_steps * disc_obs_dim) → (batch, 1)."""
    return self.disc_linear(self.disc_trunk(x))

  # ------------------------------------------------------------------
  # Normalisation helpers
  # ------------------------------------------------------------------

  def normalize_disc_obs(self, disc_obs: torch.Tensor) -> torch.Tensor:
    """Normalise per-frame.  Input: (B, disc_obs_steps, disc_obs_dim)."""
    assert disc_obs.shape[1] == self.disc_obs_steps
    assert disc_obs.shape[2] == self.disc_obs_dim
    B = disc_obs.shape[0]
    flat = disc_obs.reshape(-1, self.disc_obs_dim)
    normed = self.disc_obs_normalizer(flat)
    return normed.reshape(B, self.disc_obs_steps, self.disc_obs_dim)

  def update_normalization(self, disc_obs: torch.Tensor) -> None:
    """Update running stats.  Input: (B, disc_obs_steps, disc_obs_dim)."""
    flat = disc_obs.reshape(-1, self.disc_obs_dim)
    self.disc_obs_normalizer.update(flat)

  # ------------------------------------------------------------------
  # Gradient penalty
  # ------------------------------------------------------------------

  def compute_grad_penalty(
    self, demo_data: torch.Tensor, scale: float = 10.0
  ) -> torch.Tensor:
    """demo_data: (B, disc_obs_steps * disc_obs_dim)."""
    demo = demo_data.clone().detach().requires_grad_(True)
    disc = self.forward(demo)
    grad = autograd.grad(
      outputs=disc,
      inputs=demo,
      grad_outputs=torch.ones_like(disc),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return scale * (grad.norm(2, dim=1) ** 2).mean()

  # ------------------------------------------------------------------
  # Style reward
  # ------------------------------------------------------------------

  @torch.no_grad()
  def predict_style_reward(
    self, disc_obs: torch.Tensor, dt: float = 1.0
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute style reward for policy observations.

    Args:
        disc_obs: (num_envs, disc_obs_steps, disc_obs_dim)
        dt: environment step duration for reward scaling

    Returns:
        style_reward: (num_envs,)
        disc_score:   (num_envs,)
    """
    was_training = self.training
    self.eval()

    flat = disc_obs.view(-1, self.disc_obs_dim)
    normed = self.disc_obs_normalizer(flat)
    normed = normed.view(-1, self.disc_obs_steps * self.disc_obs_dim)
    score = self.forward(normed)  # (num_envs, 1)

    if self.loss_type == LossType.GAN:
      prob = torch.sigmoid(score)
      rew = -torch.log(torch.clamp(1.0 - prob, min=1e-6))
    elif self.loss_type == LossType.LSGAN:
      rew = torch.clamp(1.0 - 0.25 * (score - 1.0) ** 2, min=0.0)
    elif self.loss_type == LossType.WGAN:
      rew = self.disc_output_normalizer(score)
    else:
      raise ValueError(f"Unknown loss type: {self.loss_type}")

    style_reward = dt * self.style_reward_scale * rew

    if was_training:
      self.train()
      if self.loss_type == LossType.WGAN:
        self.disc_output_normalizer.update(score)

    return style_reward.squeeze(-1), score.squeeze(-1)

  def lerp_reward(
    self, task_reward: torch.Tensor, style_reward: torch.Tensor
  ) -> torch.Tensor:
    return (
      self.task_style_lerp * task_reward + (1.0 - self.task_style_lerp) * style_reward
    )
