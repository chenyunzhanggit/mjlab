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
  make_teleoperation_env_cfg,
  make_tracking_env_cfg,
)


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
) -> ManagerBasedRlEnvCfg:
  """Unitree G1 teleoperation env with an additional 'amp' observation group.

  The 'amp' group exposes raw robot state (no noise, no history) for the AMP
  discriminator:  [base_lin_vel (3), base_ang_vel (3), joint_pos (n_dof), joint_vel (n_dof)]
  """
  from mjlab.managers.observation_manager import ObservationGroupCfg

  cfg = unitree_g1_teleoperation_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  amp_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
  }
  cfg.observations["amp"] = ObservationGroupCfg(
    terms=amp_terms,
    concatenate_terms=True,
    enable_corruption=False,
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
    cfg.commands["motion"].fall_recovery_ratio = 0.0
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


def unitree_g1_student_env_cfg(play: bool = False):
  """Create the student distillation environment configuration.

  Extends ``unitree_g1_teleoperation_env_cfg`` with an additional
  ``'student'`` observation group.  The existing ``'policy'`` (teacher)
  and ``'critic'`` groups are preserved so the distillation runner can
  forward the frozen teacher through the ``'policy'`` group at every step.

  Student obs composition:
    motion_ref_vel  :  3   (current anchor linear velocity from tele-op cmd)
    hand_pos_b      :  6   (left + right wrist_yaw_link in torso frame)
    projected_gravity: 15  (history_length=5 × 3)
    base_ang_vel    : 15   (history_length=5 × 3)
    joint_pos       : 150  (history_length=5 × 30 joints)
    joint_vel       : 150  (history_length=5 × 30 joints)
    actions         : 150  (history_length=5 × 30 joints)
    ----------------------------
    Total           : 489
  """
  from mjlab.tasks.tracking.mdp import student_observations as student_obs
  from mjlab.utils.noise import UniformNoiseCfg as Unoise

  cfg = unitree_g1_teleoperation_env_cfg(has_state_estimation=False, play=play)

  student_terms = {
    # # Current-step reference anchor linear velocity: 3D
    "motion_ref_vel": ObservationTermCfg(
      func=student_obs.motion_ref_vel_current,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    # Left + right wrist_yaw_link positions in torso frame: 6D
    "hand_pos_b": ObservationTermCfg(
      func=student_obs.ref_hand_pos_b,
      params={
        "command_name": "motion",
        "hand_body_names": (
          "left_wrist_yaw_link",
          "right_wrist_yaw_link",
        ),
      },
      noise=Unoise(n_min=-0.05, n_max=0.05),
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

    ######################## all obs for test code ########################
    
    # "commands_joint_pos_only": ObservationTermCfg(
    #   func=mdp.generated_commands_joint_pos,
    #   params={"command_name": "motion"},
    #   noise=Unoise(n_min=-0.1, n_max=0.1),
    # ),
    # # ref anchor vel
    # "motion_anchor_vel_w": ObservationTermCfg(
    #   func=mdp.motion_anchor_vel_w,
    #   params={"command_name": "motion"},
    #   noise=Unoise(
    #     n_min=(-0.5, -0.5, -0.2) * 10,
    #     n_max=(0.5, 0.5, 0.2) * 10,
    #   ),
    # ),
    # # ref anchor ori  OR ref projected_gravity?
    # "motion_anchor_projected_gravity": ObservationTermCfg(
    #   func=mdp.motion_anchor_projected_gravity,
    #   params={"command_name": "motion"},
    #   noise=Unoise(n_min=-0.1, n_max=0.1),
    # ),
    # # ref anchor ang vel
    # "motion_anchor_ang_vel_w": ObservationTermCfg(
    #   func=mdp.motion_anchor_ang_vel_w,
    #   params={"command_name": "motion"},
    #   noise=Unoise(
    #     n_min=(-0.52, -0.52, -0.78) * 10,
    #     n_max=(0.52, 0.52, 0.78) * 10,
    #   ),
    # ),
    # # """ proprioceptive observations add history steps 5 frames? """
    # # proprioceptive observations
    # "projected_gravity": ObservationTermCfg(
    #   func=mdp.projected_gravity,
    #   noise=Unoise(n_min=-0.1, n_max=0.1),
    #   history_length=5,
    # ),
    # "base_ang_vel": ObservationTermCfg(
    #   func=mdp.builtin_sensor,
    #   params={"sensor_name": "robot/imu_ang_vel"},
    #   noise=Unoise(n_min=-0.2, n_max=0.2),
    #   history_length=5,
    # ),
    # "joint_pos": ObservationTermCfg(
    #   func=mdp.joint_pos_rel,
    #   noise=Unoise(n_min=-0.01, n_max=0.01),
    #   params={"biased": True},
    #   history_length=5,
    # ),
    # "joint_vel": ObservationTermCfg(
    #   func=mdp.joint_vel_rel,
    #   noise=Unoise(n_min=-1.5, n_max=1.5),
    #   history_length=5,
    # ),
    # "actions": ObservationTermCfg(
    #   func=mdp.last_action,
    #   history_length=5,
    # ),

  }

  cfg.observations["student"] = ObservationGroupCfg(
    terms=student_terms,
    concatenate_terms=True,
    enable_corruption=True,
  )

  return cfg
