"""Transformer-based ActorCritic for student policy distillation.

Architecture (actor only — critic stays MLP):
  Linear Projection → Transformer Encoder → Mean Pooling → MLP Head

Reference: the student model uses d_model=512, d_ff=1024, 2 input tokens,
4 attention heads.  The teacher remains a plain MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import MLP


class TransformerActor(nn.Module):
  """Linear projection → Transformer encoder → mean-pool → MLP head."""

  def __init__(
    self,
    input_dim: int,
    output_dim: int,
    d_model: int = 512,
    d_ff: int = 1024,
    num_tokens: int = 2,
    nhead: int = 4,
    num_layers: int = 3,
    activation: str = "gelu",
    head_hidden_dims: tuple[int, ...] = (256,),
  ):
    super().__init__()
    self.num_tokens = num_tokens
    self.d_model = d_model
    self.in_features = input_dim  # For ONNX exporter compatibility.

    # Linear projection: obs → (num_tokens, d_model)
    self.input_proj = nn.Linear(input_dim, num_tokens * d_model)

    # Learnable positional embedding
    self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)

    # Transformer encoder
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=d_ff,
      activation=activation,
      batch_first=True,
      norm_first=True,
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.norm = nn.LayerNorm(d_model)

    # MLP output head
    self.head = MLP(d_model, output_dim, list(head_hidden_dims), "elu")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    # Project and reshape to tokens: (B, num_tokens, d_model)
    tokens = self.input_proj(x).view(B, self.num_tokens, self.d_model)
    tokens = tokens + self.pos_embed
    # Transformer encoder
    tokens = self.transformer(tokens)
    tokens = self.norm(tokens)
    # Mean pooling over tokens
    pooled = tokens.mean(dim=1)  # (B, d_model)
    return self.head(pooled)

  def __getitem__(self, idx: int):
    """Compatibility shim: ONNX exporter reads ``actor[0].in_features``."""
    if idx == 0:
      return self.input_proj
    raise IndexError(idx)

  def reset(self, dones=None):
    pass

  def detach_hidden_states(self, dones=None):
    pass


class TransformerActorCritic(ActorCritic):
  """ActorCritic with a Transformer-based actor and MLP critic.

  Drop-in replacement for ``ActorCritic`` — same interface for
  ``act()``, ``act_inference()``, ``evaluate()``, normalizers, etc.
  Only ``self.actor`` is replaced; everything else is inherited.
  """

  def __init__(
    self,
    obs,
    obs_groups,
    num_actions,
    # Transformer params
    d_model: int = 512,
    d_ff: int = 1024,
    num_tokens: int = 2,
    nhead: int = 4,
    num_transformer_layers: int = 3,
    transformer_activation: str = "gelu",
    actor_head_hidden_dims: tuple[int, ...] = (256,),
    # Pass through to parent for critic, normalizers, std, etc.
    **kwargs,
  ):
    # Let the parent build the MLP actor, critic, normalizers, std, etc.
    super().__init__(obs, obs_groups, num_actions, **kwargs)

    # Now replace self.actor with the Transformer variant.
    num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])

    self.actor = TransformerActor(
      input_dim=num_actor_obs,
      output_dim=num_actions,
      d_model=d_model,
      d_ff=d_ff,
      num_tokens=num_tokens,
      nhead=nhead,
      num_layers=num_transformer_layers,
      activation=transformer_activation,
      head_hidden_dims=tuple(actor_head_hidden_dims),
    )
    print(
      f"Transformer Actor: tokens={num_tokens}, d_model={d_model}, "
      f"d_ff={d_ff}, heads={nhead}, layers={num_transformer_layers}"
    )
    total = sum(p.numel() for p in self.actor.parameters())
    print(f"  Actor params: {total:,}")
