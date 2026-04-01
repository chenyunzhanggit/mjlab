import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
    if self.alg.policy.actor_obs_normalization:
      normalizer = self.alg.policy.actor_obs_normalizer
    else:
      normalizer = None
    export_velocity_policy_as_onnx(
      self.alg.policy,
      normalizer=normalizer,
      path=policy_path,
      filename=filename,
    )
    # Attach metadata (use "local" for run_path if not using wandb)
    run_name = wandb.run.name if self.logger_type == "wandb" and wandb.run else "local"
    attach_onnx_metadata(
      self.env.unwrapped,
      run_name,  # type: ignore
      path=policy_path,
      filename=filename,
    )
    if self.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class LocoManiOnPolicyRunner(VelocityOnPolicyRunner):
  """Runner for loco-mani post-training from a distilled checkpoint.

  When ``from_distillation=True``:
  - ``load()`` resets the actor obs normalizer and action noise std.
  - ``learn()`` runs a critic-only warm-up phase (``critic_warmup_iters``
    iterations with the actor frozen) before resuming full PPO.
  """

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    from_distillation = self.cfg.get("from_distillation", False)
    warmup_iters = int(self.cfg.get("critic_warmup_iters", 0))

    if not from_distillation or warmup_iters <= 0:
      super().learn(num_learning_iterations, init_at_random_ep_len)
      return

    warmup_iters = min(warmup_iters, num_learning_iterations)
    print(
      f"[LocoManiRunner] Critic warm-up: actor frozen for {warmup_iters} iterations"
    )
    for param in self.alg.policy.actor.parameters():
      param.requires_grad_(False)

    super().learn(warmup_iters, init_at_random_ep_len)

    for param in self.alg.policy.actor.parameters():
      param.requires_grad_(True)
    print("[LocoManiRunner] Critic warm-up done — actor unfrozen")

    remaining = num_learning_iterations - warmup_iters
    if remaining > 0:
      super().learn(remaining, init_at_random_ep_len=False)

  def load(
    self, path: str, load_optimizer: bool = True, map_location: str | None = None
  ):
    from_distillation = self.cfg.get("from_distillation", False)

    if not from_distillation:
      # Normal play/resume — load everything as-is.
      return super().load(
        path, load_optimizer=load_optimizer, map_location=map_location
      )

    import math

    import torch

    # --- Distillation checkpoint resets below ---
    # Save freshly-initialised normalizer state before loading checkpoint.
    policy = self.alg.policy
    fresh_state = {
      k: v.clone()
      for k, v in policy.state_dict().items()
      if k.startswith("actor_obs_normalizer")
    }

    # Don't load the distillation optimizer — its Adam momentum is trained
    # on MSE loss and would produce destructive PPO gradient steps.
    infos = super().load(path, load_optimizer=False, map_location=map_location)

    # Restore fresh actor_obs_normalizer (wrong obs distribution —
    # motion reference vs twist/hand-pose commands).
    with torch.no_grad():
      policy.load_state_dict(fresh_state, strict=False)
    print(
      "[LocoManiRunner] Re-initialised actor_obs_normalizer (from_distillation=True)"
    )

    # Reset action noise std after loading a distilled checkpoint whose
    # std was never trained (stuck at init_noise_std=1.0).
    target_std = self.policy_cfg.get("init_noise_std", 0.05)
    with torch.no_grad():
      if hasattr(policy, "log_std"):
        policy.log_std.fill_(math.log(target_std))
        print(f"[LocoManiRunner] Reset action log_std to {target_std}")
      elif hasattr(policy, "std"):
        policy.std.fill_(target_std)
        print(f"[LocoManiRunner] Reset action std to {target_std}")
    return infos
