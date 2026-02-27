import types

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


def _compute_returns_global_norm(self, obs):
  """Override of PPO.compute_returns with cross-GPU global advantage normalization.

  In multi-GPU training each GPU holds a different shard of motion files, so
  per-GPU advantage normalization produces inconsistent gradient scales.  This
  replacement synchronises the advantage statistics across all ranks before
  normalising, so every GPU sees the same mean/std regardless of its local
  reward distribution.
  """
  last_values = self.policy.evaluate(obs).detach()
  # Compute returns/advantages without normalization first.
  self.storage.compute_returns(
    last_values, self.gamma, self.lam, normalize_advantage=False
  )
  # Per-mini-batch normalization is handled inside update(); skip here.
  if self.normalize_advantage_per_mini_batch:
    return
  adv = self.storage.advantages
  if self.is_multi_gpu:
    # Compute global mean and variance via all-reduce across all GPUs.
    local_sum = adv.sum()
    local_sq_sum = (adv**2).sum()
    local_count = torch.tensor(adv.numel(), dtype=torch.float32, device=adv.device)
    torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_sq_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_count, op=torch.distributed.ReduceOp.SUM)
    global_mean = local_sum / local_count
    global_var = (local_sq_sum / local_count - global_mean**2).clamp(min=0)
    global_std = torch.sqrt(global_var)
    self.storage.advantages = (adv - global_mean) / (global_std + 1e-8)
  else:
    # Single GPU: preserve original behaviour.
    self.storage.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  def _construct_algorithm(self, obs):
    alg = super()._construct_algorithm(obs)
    # Replace compute_returns with the global-normalisation version so that
    # advantage statistics are synchronised across GPUs before normalising.
    alg.compute_returns = types.MethodType(_compute_returns_global_norm, alg)
    return alg

  def save(self, path: str, infos=None):
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    super().save(path, infos)

  def load(
    self, path: str, load_optimizer: bool = True, map_location: str | None = None
  ):
    infos = super().load(path, load_optimizer, map_location)
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos
