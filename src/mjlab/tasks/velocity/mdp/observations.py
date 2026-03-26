from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


# ── Post-training obs (drop-in replacements for motion-reference obs) ──────
def twist_vel_xy(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """XY linear velocity from the twist command. Shape: ``(num_envs, 2)``.

  Drop-in replacement for ``student_obs.motion_ref_vel_xy`` during post-training.
  """
  from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

  cmd = env.command_manager.get_term(command_name)
  assert isinstance(cmd, UniformVelocityCommand)
  return cmd.vel_command_b[:, :2].clone()


def twist_ang_vel_z(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Yaw angular velocity from the twist command. Shape: ``(num_envs, 1)``.

  Drop-in replacement for ``student_obs.motion_ref_ang_vel_z`` during post-training.
  """
  from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

  cmd = env.command_manager.get_term(command_name)
  assert isinstance(cmd, UniformVelocityCommand)
  return cmd.vel_command_b[:, 2:3].clone()


def target_anchor_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Target torso height from the base height command. Shape: ``(num_envs, 1)``.

  Drop-in replacement for ``student_obs.ref_anchor_height`` during post-training.
  """
  from mjlab.tasks.velocity.mdp.hand_pose_command import RandomBaseHeightCommand

  cmd = env.command_manager.get_term(command_name)
  assert isinstance(cmd, RandomBaseHeightCommand)
  return cmd.target_height.clone()


def random_hand_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Random target hand positions in the torso frame. Shape: ``(num_envs, 6)``.

  Drop-in replacement for ``student_obs.motion_ref_hand_pos_b`` during post-training.
  Output layout: ``[left_xyz, right_xyz]``.
  """
  from mjlab.tasks.velocity.mdp.hand_pose_command import RandomHandPoseCommand

  cmd = env.command_manager.get_term(command_name)
  assert isinstance(cmd, RandomHandPoseCommand)
  return cmd.target_hand_pos_b.clone()


def random_hand_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Random target hand orientations (rot-mat first two cols) in torso frame.

  Drop-in replacement for ``student_obs.motion_ref_hand_ori_b`` during post-training.
  Shape: ``(num_envs, 12)``.  Layout: ``[left_6D, right_6D]``.
  """
  from mjlab.tasks.velocity.mdp.hand_pose_command import RandomHandPoseCommand

  cmd = env.command_manager.get_term(command_name)
  assert isinstance(cmd, RandomHandPoseCommand)
  return cmd.target_hand_ori_b.clone()


# ── Privileged body-state obs (no motion command required) ─────────────────


def robot_body_pos_b(
  env: ManagerBasedRlEnv,
  body_names: tuple[str, ...],
  anchor_body_name: str = "torso_link",
  entity_name: str = "robot",
) -> torch.Tensor:
  """All tracked body positions expressed in the anchor (torso) frame.

  Standalone version of the tracking-env's ``robot_body_pos_b`` — does not
  require a motion command. Shape: ``(num_envs, len(body_names) * 3)``
  """
  from mjlab.utils.lab_api.math import subtract_frame_transforms

  robot: Entity = env.scene[entity_name]
  num_bodies = len(body_names)
  body_indexes = torch.tensor(
    robot.find_bodies(body_names, preserve_order=True)[0],
    dtype=torch.long,
    device=env.device,
  )
  anchor_idx = robot.body_names.index(anchor_body_name)

  body_pos_w = robot.data.body_link_pos_w[:, body_indexes]  # (N, B, 3)
  body_quat_w = robot.data.body_link_quat_w[:, body_indexes]  # (N, B, 4)
  anchor_pos_w = robot.data.body_link_pos_w[:, anchor_idx, :][:, None, :].expand(
    -1, num_bodies, -1
  )
  anchor_quat_w = robot.data.body_link_quat_w[:, anchor_idx, :][:, None, :].expand(
    -1, num_bodies, -1
  )

  pos_b, _ = subtract_frame_transforms(
    anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w
  )
  return pos_b.reshape(env.num_envs, -1)  # (N, B*3)


def robot_body_ori_b(
  env: ManagerBasedRlEnv,
  body_names: tuple[str, ...],
  anchor_body_name: str = "torso_link",
  entity_name: str = "robot",
) -> torch.Tensor:
  """All tracked body orientations (first two rotation-matrix columns) in anchor frame.

  Standalone version of the tracking-env's ``robot_body_ori_b``.
  Shape: ``(num_envs, len(body_names) * 6)``
  """
  from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

  robot: Entity = env.scene[entity_name]
  num_bodies = len(body_names)
  body_indexes = torch.tensor(
    robot.find_bodies(body_names, preserve_order=True)[0],
    dtype=torch.long,
    device=env.device,
  )
  anchor_idx = robot.body_names.index(anchor_body_name)

  body_pos_w = robot.data.body_link_pos_w[:, body_indexes]  # (N, B, 3)
  body_quat_w = robot.data.body_link_quat_w[:, body_indexes]  # (N, B, 4)
  anchor_pos_w = robot.data.body_link_pos_w[:, anchor_idx, :][:, None, :].expand(
    -1, num_bodies, -1
  )
  anchor_quat_w = robot.data.body_link_quat_w[:, anchor_idx, :][:, None, :].expand(
    -1, num_bodies, -1
  )

  _, ori_b = subtract_frame_transforms(
    anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w
  )
  mat = matrix_from_quat(ori_b)  # (N, B, 3, 3)
  return mat[..., :2].reshape(env.num_envs, -1)  # (N, B*6)


def robot_base_height(
  env: ManagerBasedRlEnv,
  entity_name: str = "robot",
  body_name: str = "pelvis",
) -> torch.Tensor:
  """Height of a body above the world ground plane. Shape: ``(num_envs, 1)``."""
  robot: Entity = env.scene[entity_name]
  body_idx = robot.body_names.index(body_name)
  return robot.data.body_link_pos_w[:, body_idx, 2:3]  # (N, 1)
