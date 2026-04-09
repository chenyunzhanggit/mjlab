"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_post_train_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """RL runner config matching the distilled student model architecture."""
  return RslRlOnPolicyRunnerCfg(
    # Actor reads the "student" obs group (475D with history);
    # critic reads the "critic" obs group (107D).
    obs_groups={"policy": ("student",), "critic": ("critic",)},
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=0.5,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(1024, 1024, 512, 512),
      critic_hidden_dims=(2048, 2048, 1024, 1024, 512, 512),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.001,
      num_learning_epochs=3,
      num_mini_batches=4,
      learning_rate=1.0e-5,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.005,
      max_grad_norm=0.5,
    ),
    experiment_name="g1_loco_mani",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
    critic_warmup_iters=500,
  )


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
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
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
