"""Student fine-tuning runner for motion-tracking tasks.

Phase-3 training pipeline:
  1. Load student actor + actor_obs_normalizer from distillation checkpoint.
  2. Load teacher critic + critic_obs_normalizer from teacher checkpoint.
  3. Reset action noise std.
  4. Optional critic warm-up (actor frozen for first N iterations).
  5. Full PPO fine-tune on the tracking environment.
"""

from __future__ import annotations

import math

import torch

from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner


class StudentTrackingFineTuneRunner(MotionTrackingOnPolicyRunner):
  """PPO fine-tuning runner that bootstraps actor from distillation and
  critic from the teacher checkpoint.

  When ``from_distillation=True`` in the runner config:
  - Actor weights (+ actor_obs_normalizer) come from the student distillation
    checkpoint loaded via the standard resume mechanism.
  - Critic weights (+ critic_obs_normalizer) are overwritten with the
    teacher checkpoint specified by ``teacher_checkpoint``.
  - Action noise std is reset to ``init_noise_std``.
  - An optional critic warm-up phase (``critic_warmup_iters``) keeps the
    actor frozen so the critic can adapt before the actor starts moving.
  """

  def load(
    self, path: str, load_optimizer: bool = True, map_location: str | None = None
  ):
    from_distillation = self.cfg.get("from_distillation", False)

    if not from_distillation:
      return super().load(
        path, load_optimizer=load_optimizer, map_location=map_location
      )

    # Step 1: manually load only actor + actor_obs_normalizer from the
    # distillation checkpoint, skipping critic (architecture differs).
    map_loc = map_location or self.device
    ckpt = torch.load(path, map_location=map_loc, weights_only=False)
    actor_state = {
      k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("actor")
    }
    self.alg.policy.load_state_dict(actor_state, strict=False)
    infos = ckpt.get("infos", None)
    print(
      f"[StudentFineTuneRunner] Loaded actor + actor_obs_normalizer from: {path}  "
      f"(iter {ckpt.get('iter', '?')})"
    )

    policy = self.alg.policy

    # Step 2: critic starts from random init (student env critic obs = 119-dim,
    # incompatible with teacher's 841-dim).  critic_warmup_iters bootstraps it.
    print("[StudentFineTuneRunner] Critic initialised from scratch.")

    # Step 3: reset action std (distillation never trained std).
    target_std = self.policy_cfg.get("init_noise_std", 0.60)
    with torch.no_grad():
      if hasattr(policy, "log_std"):
        policy.log_std.fill_(math.log(target_std))
        print(f"[StudentFineTuneRunner] Reset log_std → {target_std}")
      elif hasattr(policy, "std"):
        policy.std.fill_(target_std)
        print(f"[StudentFineTuneRunner] Reset std → {target_std}")

    return infos

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
      f"[StudentFineTuneRunner] Critic warm-up: actor frozen for {warmup_iters} iterations"
    )
    for param in self.alg.policy.actor.parameters():
      param.requires_grad_(False)

    super().learn(warmup_iters, init_at_random_ep_len)

    for param in self.alg.policy.actor.parameters():
      param.requires_grad_(True)

    # Reset Adam state for actor parameters.  During warm-up they had
    # requires_grad=False so optimizer.step() skipped them entirely —
    # their (m, v, step) stayed at zero.  On the first real update the
    # effective lr would be lr / sqrt(eps) ≈ lr * 1e4, causing immediate
    # action explosion.  Clearing the state lets Adam ramp up naturally.
    actor_param_ids = {id(p) for p in self.alg.policy.actor.parameters()}
    for p, _state in list(self.alg.optimizer.state.items()):
      if id(p) in actor_param_ids:
        self.alg.optimizer.state[p] = {}
    print(
      "[StudentFineTuneRunner] Critic warm-up done — actor unfrozen, Adam state reset"
    )

    # Soft-unfreeze transition: run a short phase with a reduced lr so the
    # actor warms up gradually before full-speed PPO.
    transition_iters = int(self.cfg.get("actor_unfreeze_transition_iters", 200))
    remaining = num_learning_iterations - warmup_iters
    transition_iters = min(transition_iters, remaining)
    if transition_iters > 0:
      normal_lr = float(self.cfg.get("learning_rate", 1e-5))
      transition_lr = normal_lr * 0.1
      for param_group in self.alg.optimizer.param_groups:
        param_group["lr"] = transition_lr
      print(
        f"[StudentFineTuneRunner] Soft-unfreeze transition: {transition_iters} iters "
        f"at lr={transition_lr:.2e}"
      )
      super().learn(transition_iters, init_at_random_ep_len=False)
      for param_group in self.alg.optimizer.param_groups:
        param_group["lr"] = normal_lr
      print(f"[StudentFineTuneRunner] Transition done — lr restored to {normal_lr:.2e}")
      remaining -= transition_iters

    if remaining > 0:
      super().learn(remaining, init_at_random_ep_len=False)
