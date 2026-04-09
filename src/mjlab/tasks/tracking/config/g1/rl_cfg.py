"""RL configuration for Unitree G1 tracking task."""

from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_tracking",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_teleoperation_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      # class_name="ActorCritic", #"rsl_rl.modules.MjlabActorCritic",
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(2048, 2048, 1024, 1024, 512, 512),
      critic_hidden_dims=(2048, 2048, 1024, 1024, 512, 512),
      # actor_hidden_dims=(2048, 1024, 512),
      # critic_hidden_dims=(2048, 1024, 512),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.013,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_teleoperation",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )


def unitree_g1_teacher_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 tracking task."""
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      # class_name="ActorCritic", #"rsl_rl.modules.MjlabActorCritic",
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(2048, 1024, 1024, 512, 512),
      critic_hidden_dims=(2048, 1024, 1024, 512, 512),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.013,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_teacher",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )


# ---------------------------------------------------------------------------
# AMP configuration
# ---------------------------------------------------------------------------


@dataclass
class AmpCfg:
  """Hyperparameters for AMP discriminator training.

  Demo observations are sampled from the MultiMotionLoader that is already
  loaded into GPU memory by MultiMotionCommand — no extra data loading.

  ``disc_obs_steps`` controls how many consecutive frames are stacked before
  being fed to the discriminator.  Set to 1 for single-frame AMP (original
  behaviour); larger values give the discriminator temporal context.

  ``disc_obs_dim`` and ``step_dt`` are resolved automatically by the runner
  from the environment and should not be set manually.
  """

  # ── Discriminator architecture ─────────────────────────────────────
  disc_hidden_dims: tuple[int, ...] = (1024, 512, 512, 256)
  """Hidden layer sizes for the discriminator MLP."""

  disc_activation: str = "elu"
  """Activation function for the discriminator trunk."""

  # ── Loss type ──────────────────────────────────────────────────────
  loss_type: str = "LSGAN"
  """Discriminator loss: 'GAN', 'LSGAN', or 'WGAN'."""

  # ── Discriminator training ─────────────────────────────────────────
  disc_learning_rate: float = 1e-4
  disc_trunk_weight_decay: float = 1e-4
  """L2 regularisation for trunk parameters."""
  disc_linear_weight_decay: float = 1e-2
  """L2 regularisation for the output linear layer."""
  disc_max_grad_norm: float = 1.0
  """Max gradient norm for discriminator clipping."""

  grad_penalty_scale: float = 10.0
  """Gradient penalty coefficient (0 to disable)."""

  # ── Buffer ─────────────────────────────────────────────────────────
  disc_obs_buffer_size: int = 1000
  """Circular buffer capacity (in env steps) for policy/demo AMP observations."""

  # ── Reward mixing ──────────────────────────────────────────────────
  task_style_lerp: float = 0.3
  """Reward interpolation: 1.0 = pure task reward, 0.0 = pure style reward."""

  style_reward_scale: float = 2.0
  """Scale applied to the discriminator style reward."""

  # ── History ────────────────────────────────────────────────────────
  disc_obs_steps: int = 2
  """Number of consecutive observation frames stacked for the discriminator.
  Must match the ``history_length`` set in the 'amp' observation group of
  the environment config.  Set to 1 for single-frame (no history) AMP."""

  # ── IMU body ───────────────────────────────────────────────────────
  imu_body_name: str = ""
  """MuJoCo body name where the IMU sensor is mounted.
  Demo obs velocities are rotated into this body's frame so they match the
  env's ``robot/imu_lin_vel`` / ``robot/imu_ang_vel`` sensor outputs.
  Leave empty to fall back to ``anchor_body_name`` (not recommended)."""

  # ── Filled in by the runner — do not set manually ──────────────────
  disc_obs_dim: int = 0
  """Observation dimension per frame (resolved from env at runtime)."""

  step_dt: float = 0.02
  """Environment step duration in seconds (resolved from env at runtime)."""


@dataclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config that extends the standard PPO config with AMP fields."""

  amp: AmpCfg = field(default_factory=AmpCfg)


def unitree_g1_teleoperation_amp_runner_cfg() -> RslRlAmpRunnerCfg:
  """Create AMP runner configuration for Unitree G1 teleoperation task."""
  base = unitree_g1_teleoperation_ppo_runner_cfg()
  return RslRlAmpRunnerCfg(
    policy=base.policy,
    algorithm=base.algorithm,
    experiment_name="g1_teleoperation_amp",
    save_interval=2000,
    num_steps_per_env=base.num_steps_per_env,
    max_iterations=base.max_iterations,
    # G1 IMU is mounted in the pelvis body (site imu_in_pelvis in g1.xml).
    # Demo obs velocities must be expressed in the same frame.
    amp=AmpCfg(imu_body_name="pelvis", disc_obs_steps=5, disc_obs_buffer_size=100),
  )


# ---------------------------------------------------------------------------
# Distillation configuration
# ---------------------------------------------------------------------------


@dataclass
class DistillationPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner configuration for Phase-2 knowledge-distillation training.

  Extends :class:`RslRlOnPolicyRunnerCfg` with distillation-specific
  hyper-parameters that are passed through to
  :class:`MotionTrackingDistillationRunner` via the ``train_cfg`` dict.
  """

  # ── Teacher ────────────────────────────────────────────────────────────
  teacher_checkpoint: str = ""
  """Absolute path to the teacher model checkpoint (``.pt`` file).
    Must be set before training starts."""

  teacher_actor_hidden_dims: tuple[int, ...] = (2048, 1024, 1024, 512, 512)
  """Hidden-layer sizes of the *teacher* actor MLP.
    Must match the architecture used when training the teacher."""

  # ── Distillation loss ──────────────────────────────────────────────────
  distill_coef: float = 1.0
  """Initial weight applied to the distillation MSE loss term."""

  distill_coef_decay: float = 0.9999
  """Multiplicative decay factor applied to ``distill_coef`` after each
    training iteration.  Set to 1.0 to keep the coefficient constant."""

  distill_epochs: int = 2
  """Number of gradient-update epochs over the collected rollout buffer
    per training iteration (distillation phase only)."""

  distill_num_mini_batches: int = 4
  """Number of mini-batches per distillation epoch."""


@dataclass
class StudentFineTunePpoRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config for Phase-3 student PPO fine-tuning on the tracking task.

  Actor is loaded from a distillation checkpoint; critic is bootstrapped from
  the teacher checkpoint.  Set ``from_distillation=True`` and provide both
  ``resume=True`` (for the student actor) and ``teacher_checkpoint`` (for the
  critic) when launching training.
  """

  teacher_checkpoint: str = ""
  """Absolute path to the teacher .pt checkpoint used to initialise the critic."""


def unitree_g1_student_finetune_runner_cfg() -> StudentFineTunePpoRunnerCfg:
  """RL runner config for Phase-3 student PPO fine-tuning."""
  return StudentFineTunePpoRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=0.1,
      # Distillation ran under inference_mode so actor_obs_normalizer was never
      # updated (count stays 0).  The actor learned with raw unnormalised obs.
      # Enabling normalisation here would shift the input distribution and
      # degrade the loaded policy.  Keep it off to match distillation behaviour.
      actor_obs_normalization=False,
      critic_obs_normalization=True,
      # Actor matches distilled student architecture.
      actor_hidden_dims=(1024, 1024, 512, 512),
      # Critic reads the 119-dim student critic obs group (not teacher's 841-dim).
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1e-5,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=0.5,
    ),
    obs_groups={"policy": ("student",), "critic": ("critic",)},
    experiment_name="g1_student_finetune",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=50_000,
    from_distillation=True,
    critic_warmup_iters=0,
    teacher_checkpoint="",
  )


def unitree_g1_student_distill_runner_cfg() -> DistillationPpoRunnerCfg:
  """Create RL runner configuration for Unitree G1 student distillation."""
  return DistillationPpoRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      # Student actor is smaller than teacher (1024 × 4 vs 2048 × 6).
      actor_hidden_dims=(1024, 1024, 512, 512),
      # Critic keeps full capacity — it still sees privileged obs.
      critic_hidden_dims=(2048, 2048, 1024, 1024, 512, 512),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    # Student actor reads from the 'student' obs group;
    # critic still reads full privileged 'critic' obs.
    obs_groups={"policy": ("student",), "critic": ("critic",)},
    experiment_name="g1_student",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=50_000,
    teacher_checkpoint="",
    teacher_actor_hidden_dims=(2048, 1024, 1024, 512, 512),
    distill_coef=1.0,
    distill_coef_decay=0.9999,
    distill_epochs=2,
    distill_num_mini_batches=4,
  )
