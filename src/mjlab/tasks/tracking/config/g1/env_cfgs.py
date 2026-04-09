"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg, MultiMotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import (
  make_teacher_env_cfg,
  make_teleoperation_env_cfg,
  make_tracking_env_cfg,
)
from mjlab.tasks.velocity.mdp.observations import robot_base_height


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_policy_terms = {
      k: v
      for k, v in cfg.observations["policy"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["policy"] = ObservationGroupCfg(
      terms=new_policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


def unitree_g1_teleoperation_amp_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  disc_obs_steps: int = 2,
) -> ManagerBasedRlEnvCfg:
  """Unitree G1 teleoperation env with an additional 'amp' observation group.

  The 'amp' group exposes raw robot state for the AMP discriminator with
  temporal history:
    [joint_pos (n_dof)]
  per frame, stacked over ``disc_obs_steps`` consecutive steps.

  obs["amp"] shape: (num_envs, disc_obs_steps, obs_dim_per_frame)

  Args:
    disc_obs_steps: Number of consecutive frames for the discriminator.
      Must match ``AmpCfg.disc_obs_steps`` in the runner config.
      Set to 1 for single-frame (no history) AMP.
  """
  from mjlab.managers.observation_manager import ObservationGroupCfg

  cfg = unitree_g1_teleoperation_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  amp_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "joint_pos": ObservationTermCfg(func=mdp.amp_abs_joint_pos),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
  }
  # history_length set at group level so flatten_history_dim=False propagates to
  # all terms. Each term → (num_envs, disc_obs_steps, dim_i); concatenate along
  # dim=-1 → final shape: (num_envs, disc_obs_steps, sum_dim).
  cfg.observations["amp"] = ObservationGroupCfg(
    terms=amp_terms,
    concatenate_terms=True,
    enable_corruption=False,
    history_length=disc_obs_steps,
    flatten_history_dim=False,
  )
  return cfg


def unitree_g1_teleoperation_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_teleoperation_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]

  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_policy_terms = {
      k: v
      for k, v in cfg.observations["policy"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["policy"] = ObservationGroupCfg(
      terms=new_policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)
    motion_cmd.fall_recovery_ratio = 0.0
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


def unitree_g1_teacher_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_teacher_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]

  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_policy_terms = {
      k: v
      for k, v in cfg.observations["policy"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["policy"] = ObservationGroupCfg(
      terms=new_policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)
    motion_cmd.fall_recovery_ratio = 0.0
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


def unitree_g1_student_env_cfg(play: bool = False):
  """Create the student distillation environment configuration.

  Extends ``unitree_g1_teacher_env_cfg`` with an additional
  ``'student'`` observation group.  The existing ``'policy'`` (teacher)
  and ``'critic'`` groups are preserved so the distillation runner can
  forward the frozen teacher through the ``'policy'`` group at every step.

  """
  from mjlab.tasks.tracking.mdp import student_observations as student_obs
  from mjlab.utils.noise import UniformNoiseCfg as Unoise

  cfg = unitree_g1_teacher_env_cfg(has_state_estimation=False, play=play)

  student_terms = {
    # # Current-step reference anchor linear velocity: 3D
    "motion_ref_vel": ObservationTermCfg(
      func=student_obs.motion_ref_vel_xy,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "motion_ref_ang_vel": ObservationTermCfg(
      func=student_obs.motion_ref_ang_vel_z,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "motion_ref_anchor_height": ObservationTermCfg(
      func=student_obs.ref_anchor_height,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    # Left + right wrist_yaw_link positions in torso frame: 6D
    "hand_pos_b": ObservationTermCfg(
      func=student_obs.motion_ref_hand_pos_b,
      params={
        "command_name": "motion",
        "hand_body_names": (
          "left_wrist_yaw_link",
          "right_wrist_yaw_link",
        ),
      },
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    # Left + right wrist_yaw_link orientations in torso frame: 12D
    "hand_ori_b": ObservationTermCfg(
      func=student_obs.motion_ref_hand_ori_b,
      params={
        "command_name": "motion",
        "hand_body_names": (
          "left_wrist_yaw_link",
          "right_wrist_yaw_link",
        ),
      },
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    # IMU gravity orientation with 5-step history: 15D
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.1, n_max=0.1),
      history_length=5,
    ),
    # IMU angular velocity with 5-step history: 15D
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      history_length=5,
    ),
    # Joint positions (encoder) with 5-step history: 150D
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
      history_length=5,
    ),
    # Joint velocities with 5-step history: 150D
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      history_length=5,
    ),
    # Previous actions with 5-step history: 150D
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      history_length=5,
    ),
  }

  cfg.observations["student"] = ObservationGroupCfg(
    terms=student_terms,
    concatenate_terms=True,
    enable_corruption=True,
  )

  # ── Override critic to match post-training structure ──────────────────────
  critic_terms = {
    # Task commands (2D + 1D + 1D + 6D = 10D)
    "motion_ref_vel": ObservationTermCfg(
      func=student_obs.motion_ref_vel_xy,
      params={"command_name": "motion"},
    ),  # 2D
    "motion_ref_ang_vel": ObservationTermCfg(
      func=student_obs.motion_ref_ang_vel_z,
      params={"command_name": "motion"},
    ),  # 1D
    "height_cmd": ObservationTermCfg(
      func=student_obs.ref_anchor_height,
      params={"command_name": "motion"},
    ),  # 1D
    "hand_pos_cmd": ObservationTermCfg(
      func=student_obs.motion_ref_hand_pos_b,
      params={
        "command_name": "motion",
        "hand_body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
      },
    ),  # 6D
    "hand_ori_cmd": ObservationTermCfg(
      func=student_obs.motion_ref_hand_ori_b,
      params={
        "command_name": "motion",
        "hand_body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
      },
    ),  # 12D
    # Pelvis height (1D)
    "base_height": ObservationTermCfg(func=robot_base_height),  # 1D
    # Proprioception (3D + 3D + 3D + 29D + 29D + 29D = 96D)
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),  # 3D
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )

  # ── Rewards (explicit copy from make_teacher_env_cfg for easy tuning) ────
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  cfg.rewards = {
    # ── Root stability ────────────────────────────────────────────────────
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=1.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp_fall_recovery,
      weight=1.2,
      params={"command_name": "motion", "std": 0.4},
    ),
    # "anchor_height": RewardTermCfg(
    #   func=mdp.motion_global_anchor_height_error_exp_fall_recovery,
    #   weight=2.0,
    #   params={"command_name": "motion", "std": 0.3},
    # ),
    "base_height": RewardTermCfg(
      func=mdp.motion_global_anchor_height_error_exp,
      weight=0.75,
      params={"command_name": "motion", "std": 0.1},
    ),
    # ── Full-body tracking (auxiliary, low weight) ────────────────────────
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp_fall_recovery,
      weight=0.3,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp_fall_recovery,
      weight=0.3,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp_fall_recovery,
      weight=0.3,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp_fall_recovery,
      weight=0.3,
      params={"command_name": "motion", "std": 3.14},
    ),
    # ── Pelvis velocity (primary focus) ───────────────────────────────────
    "pelvis_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp_fall_recovery,
      weight=2.0,
      params={"command_name": "motion", "std": 0.5, "body_names": ("pelvis",)},
    ),
    "pelvis_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp_fall_recovery,
      weight=1.25,
      params={"command_name": "motion", "std": 0.5, "body_names": ("pelvis",)},
    ),
    # ── Wrist tracking (primary focus) ───────────────────────────────────
    # "wrist_pos": RewardTermCfg(
    #   func=mdp.motion_relative_body_position_error_exp_fall_recovery,
    #   weight=2.0,
    #   params={
    #     "command_name": "motion",
    #     "std": 0.15,
    #     "body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    #   },
    # ),
    "global_wrist_pos": RewardTermCfg(
      func=mdp.motion_global_body_position_error_exp,
      weight=2.0,
      params={
        "command_name": "motion",
        "std": 0.3,
        "body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
      },
    ),
    "wrist_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.2,
      params={
        "command_name": "motion",
        "std": 0.3,
        "body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
      },
    ),
    # ── Regularisation ────────────────────────────────────────────────────
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1.0e-1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision"},
    ),
  }

  return cfg
