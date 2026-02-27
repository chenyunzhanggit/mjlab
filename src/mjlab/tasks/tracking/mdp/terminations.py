from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MotionCommand
from .multi_commands import MultiMotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)


def bad_anchor_ori_fall_recovery(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  termination_mask = (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold
  # Check if this is a MultiMotionCommand with fall recovery support
  if isinstance(command, MultiMotionCommand):
    # Get fall recovery mask
    fall_recovery_mask = getattr(command, "init_fall_recovery_mask", None)
    if fall_recovery_mask is not None:
      current_steps = command.time_steps - command.buffer_start_time
      # 150 = 3s / 0.02s (step_dt)
      fall_recovery_protected = fall_recovery_mask & (current_steps < 150)
      # Set termination to False for protected fall recovery environments
      termination_mask = termination_mask & ~fall_recovery_protected

  return termination_mask


def bad_motion_body_pos_z_only_fall_recovery(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  termination_mask = torch.any(error > threshold, dim=-1)
  # Check if this is a MultiMotionCommand with fall recovery support
  if isinstance(command, MultiMotionCommand):
    # Get fall recovery mask
    fall_recovery_mask = getattr(command, "init_fall_recovery_mask", None)
    if fall_recovery_mask is not None:
      current_steps = command.time_steps - command.buffer_start_time
      # 150 = 3s / 0.02s (step_dt)
      fall_recovery_protected = fall_recovery_mask & (current_steps < 150)
      # Set termination to False for protected fall recovery environments
      termination_mask = termination_mask & ~fall_recovery_protected

  return termination_mask


def bad_anchor_pos_z_only_fall_recovery(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  termination_mask = (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )
  # Check if this is a MultiMotionCommand with fall recovery support
  if isinstance(command, MultiMotionCommand):
    # Get fall recovery mask
    fall_recovery_mask = getattr(command, "init_fall_recovery_mask", None)
    if fall_recovery_mask is not None:
      current_steps = command.time_steps - command.buffer_start_time
      # 150 = 3s / 0.02s (step_dt)
      fall_recovery_protected = fall_recovery_mask & (current_steps < 150)
      # Set termination to False for protected fall recovery environments
      termination_mask = termination_mask & ~fall_recovery_protected

  return termination_mask
