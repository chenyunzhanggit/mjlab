"""Unitree G1 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def unitree_g1_post_train_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Post-training env for the distilled student policy on flat terrain.

  The student obs interface is preserved exactly but motion-reference observations
  are replaced with randomly sampled commands:

  * ``motion_ref_vel``         ← xy velocity from the ``twist`` command (2D)
  * ``motion_ref_ang_vel``     ← yaw rate from the ``twist`` command (1D)
  * ``motion_ref_anchor_height`` ← nominal torso height from ``hand_pose`` cmd (1D)
  * ``hand_pos_b``             ← random target hand positions from ``hand_pose`` cmd (6D)

  The remaining obs (projected_gravity, base_ang_vel, joint_pos, joint_vel, actions)
  are identical to the student training env (with the same history lengths / noise).
  """
  from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
  from mjlab.tasks.velocity.mdp import (
    RandomBaseHeightCommandCfg,
    RandomHandPoseCommandCfg,
  )
  from mjlab.utils.noise import UniformNoiseCfg as Unoise

  cfg = unitree_g1_flat_env_cfg(play=play)

  # ── Add hand pose command ────────────────────────────────────────────────
  cfg.commands["hand_pose"] = RandomHandPoseCommandCfg(
    resampling_time_range=(4.0, 7.0),
    debug_vis=True,
    ranges=RandomHandPoseCommandCfg.Ranges(
      left_x=(0.0, 0.28),
      left_y=(0.0, 0.40),
      left_z=(0.0, 0.45),
      right_x=(0.0, 0.28),
      right_y=(-0.40, 0.0),
      right_z=(0.0, 0.45),
    ),
  )

  # ── Add base height command ──────────────────────────────────────────────
  cfg.commands["base_height"] = RandomBaseHeightCommandCfg(
    resampling_time_range=(4.0, 7.0),
    randomize=True,
    height_range=(0.50, 0.80),
  )

  # ── Student policy obs group ─────────────────────────────────────────────
  # Obs names and dims match the distilled student policy exactly.
  student_terms = {
    # velocity commands replacing motion reference (2D + 1D + 1D = 4D)
    "cmd_lin_xy": ObservationTermCfg(
      func=mdp.twist_vel_xy,
      params={"command_name": "twist"},
    ),
    "cmd_ang_z": ObservationTermCfg(
      func=mdp.twist_ang_vel_z,
      params={"command_name": "twist"},
    ),
    "cmd_base_height": ObservationTermCfg(
      func=mdp.target_anchor_height,
      params={"command_name": "base_height"},
      noise=Unoise(n_min=-0.04, n_max=0.04),
    ),
    # random hand targets replacing motion reference hand pos (6D)
    "cmd_hand_pos_b": ObservationTermCfg(
      func=mdp.random_hand_pos_b,
      params={"command_name": "hand_pose"},
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    # random hand targets replacing motion reference hand ori (12D)
    # "cmd_hand_ori_b": ObservationTermCfg(
    #   func=mdp.random_hand_ori_b,
    #   params={"command_name": "hand_pose"},
    #   noise=Unoise(n_min=-0.02, n_max=0.02),
    # ),
    # proprioception with 5-step history — identical to student training
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.1, n_max=0.1),
      history_length=5,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      history_length=5,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
      history_length=5,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      history_length=5,
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      history_length=5,
    ),
  }

  cfg.observations["student"] = ObservationGroupCfg(
    terms=student_terms,
    concatenate_terms=True,
    enable_corruption=not play,
  )

  # ── Critic obs: task-command + privileged body state ─────────────────────
  # Dim = 2 + 1 + 6 + 1 + 1 + 3 + 3 + 3 + 29 + 29 + 29 = 107D
  # Matches the distillation critic structure so weights transfer cleanly.
  critic_terms = {
    # Task commands (2D + 1D + 6D + 1D = 10D)
    "cmd_lin_xy": ObservationTermCfg(
      func=mdp.twist_vel_xy, params={"command_name": "twist"}
    ),  # 2D
    "cmd_ang_z": ObservationTermCfg(
      func=mdp.twist_ang_vel_z, params={"command_name": "twist"}
    ),  # 1D
    "cmd_base_height": ObservationTermCfg(
      func=mdp.target_anchor_height, params={"command_name": "base_height"}
    ),  # 1D
    "cmd_hand_pos_b": ObservationTermCfg(
      func=mdp.random_hand_pos_b, params={"command_name": "hand_pose"}
    ),  # 6D
    # "cmd_hand_ori_b": ObservationTermCfg(
    #   func=mdp.random_hand_ori_b, params={"command_name": "hand_pose"}
    # ),  # 12D
    # Pelvis height (1D)
    "base_height": ObservationTermCfg(func=mdp.robot_base_height),  # 1D
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

  # ── Loco-mani tracking rewards ───────────────────────────────────────────
  cfg.rewards["track_base_height"] = RewardTermCfg(
    func=mdp.track_base_height,
    weight=1.0,
    params={
      "std": 0.1,
      "command_name": "base_height",
      "body_name": "pelvis",
    },
  )
  cfg.rewards["track_ee_pos"] = RewardTermCfg(
    func=mdp.track_ee_pos,
    weight=1.0,
    params={
      "std": 0.2,
      "command_name": "hand_pose",
      "anchor_body_name": "torso_link",
      "hand_body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    },
  )
  cfg.rewards["track_ee_ori"] = RewardTermCfg(
    func=mdp.track_ee_ori,
    weight=0.0,
    params={
      "std": 0.3,
      "command_name": "hand_pose",
      "anchor_body_name": "torso_link",
      "hand_body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    },
  )

  return cfg
