"""Student policy observation functions for knowledge distillation.

These observations contain only information available during real-time
teleoperation deployment — no privileged future reference frames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.tracking.mdp.multi_commands import MultiMotionCommand
from mjlab.utils.lab_api.math import subtract_frame_transforms

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_ref_vel_current(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Current-timestep reference anchor linear velocity (no history or future).

  Unlike ``motion_anchor_vel_w`` which returns
  [history_steps + future_steps] frames, this function extracts only the
  single current-timestep frame so it is directly available during
  real-time teleoperation (the operator sends the current target velocity).

  Shape: ``(num_envs, 3)``
  """
  command = cast(MultiMotionCommand, env.command_manager.get_term(command_name))
  # anchor_lin_vel_w has shape (num_envs, total_steps * 3)
  # where total_steps = history_steps + future_steps
  # Current step sits at index history_steps in the time dimension.
  vel_all = command.anchor_lin_vel_w
  num_steps = command.cfg.history_steps + command.cfg.future_steps
  vel_reshaped = vel_all.view(env.num_envs, num_steps, 3)
  current_idx = command.cfg.history_steps
  return vel_reshaped[:, current_idx, :].contiguous()  # (num_envs, 3)

def motion_ref_ang_vel_current(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Current-timestep reference anchor angular velocity (no history or future).

  Unlike ``motion_anchor_vel_w`` which returns
  [history_steps + future_steps] frames, this function extracts only the
  single current-timestep frame so it is directly available during
  real-time teleoperation (the operator sends the current target velocity).

  Shape: ``(num_envs, 3)``
  """
  command = cast(MultiMotionCommand, env.command_manager.get_term(command_name))
  # anchor_ang_vel_w has shape (num_envs, total_steps * 3)
  # where total_steps = history_steps + future_steps
  # Current step sits at index history_steps in the time dimension.
  ang_vel_all = command.anchor_ang_vel_w
  num_steps = command.cfg.history_steps + command.cfg.future_steps
  ang_vel_reshaped = ang_vel_all.view(env.num_envs, num_steps, 3)
  current_idx = command.cfg.history_steps
  return ang_vel_reshaped[:, current_idx, :].contiguous()  # (num_envs, 3)

def ref_anchor_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor (torso) height above the ground plane.

  Returns a single scalar per environment, representing the current
  reference torso height. This is directly available during real-time
  teleoperation since the operator can observe the current torso height.

  Shape: ``(num_envs, 1)``
  """
  command = cast(MultiMotionCommand, env.command_manager.get_term(command_name))
  # Robot anchor (torso) position in world frame: (N, 3)
  anchor_pos_w = command.robot_anchor_pos_w
  # Ground plane is assumed to be at z=0 in world frame, so height is just z-coordinate.
  anchor_height = anchor_pos_w[:, 2:3]  # (N, 1)
  return anchor_height.contiguous()


def motion_ref_hand_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  hand_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  ),
) -> torch.Tensor:
  """Reference motion hand positions expressed in the reference anchor frame.

  Uses the current-timestep reference motion state (not the actual robot state),
  so this observation is only available during training (privileged).
  Each hand position is a 3-vector in the reference anchor frame, giving a total
  output of ``len(hand_body_names) * 3`` dimensions.

  Default: left + right ``wrist_yaw_link`` → shape ``(num_envs, 6)``
  """
  command = cast(MultiMotionCommand, env.command_manager.get_term(command_name))

  num_hands = len(hand_body_names)

  # Resolve hand indices within cfg.body_names (the tracked body list).
  hand_idx = torch.tensor(
    [command.cfg.body_names.index(name) for name in hand_body_names],
    dtype=torch.long,
    device=env.device,
  )

  # Reference body poses in world frame: body_pos_w does NOT include env_origins,
  # so we read anchor from the same buffer to keep both in the same frame.
  ref_body_pos_w = command.body_pos_w  # (N, num_bodies, 3)
  ref_body_quat_w = command.body_quat_w  # (N, num_bodies, 4)

  anchor_idx = command.motion_anchor_body_index
  anchor_pos_w = ref_body_pos_w[:, anchor_idx, :][:, None, :].expand(-1, num_hands, -1)
  anchor_quat_w = ref_body_quat_w[:, anchor_idx, :][:, None, :].expand(
    -1, num_hands, -1
  )

  hand_pos_w = ref_body_pos_w[:, hand_idx, :]  # (N, num_hands, 3)
  hand_quat_w = ref_body_quat_w[:, hand_idx, :]  # (N, num_hands, 4)

  pos_b, _ = subtract_frame_transforms(
    anchor_pos_w,
    anchor_quat_w,
    hand_pos_w,
    hand_quat_w,
  )

  return pos_b.reshape(env.num_envs, -1)  # (N, num_hands * 3)


def motion_ref_foot_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  foot_body_names: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ),
) -> torch.Tensor:
  """Reference motion foot positions expressed in the reference anchor frame.

  Uses the current-timestep reference motion state (not the actual robot state),
  so this observation is only available during training (privileged).
  Each foot position is a 3-vector in the reference anchor frame, giving a total
  output of ``len(foot_body_names) * 3`` dimensions.

  Default: left + right ``ankle_roll_link`` → shape ``(num_envs, 6)``
  """
  command = cast(MultiMotionCommand, env.command_manager.get_term(command_name))

  num_feet = len(foot_body_names)

  # Resolve foot indices within cfg.body_names (the tracked body list).
  foot_idx = torch.tensor(
    [command.cfg.body_names.index(name) for name in foot_body_names],
    dtype=torch.long,
    device=env.device,
  )

  # Reference body poses in world frame: body_pos_w does NOT include env_origins,
  # so we read anchor from the same buffer to keep both in the same frame.
  ref_body_pos_w = command.body_pos_w  # (N, num_bodies, 3)
  ref_body_quat_w = command.body_quat_w  # (N, num_bodies, 4)

  anchor_idx = command.motion_anchor_body_index
  anchor_pos_w = ref_body_pos_w[:, anchor_idx, :][:, None, :].expand(-1, num_feet, -1)
  anchor_quat_w = ref_body_quat_w[:, anchor_idx, :][:, None, :].expand(-1, num_feet, -1)

  foot_pos_w = ref_body_pos_w[:, foot_idx, :]  # (N, num_feet, 3)
  foot_quat_w = ref_body_quat_w[:, foot_idx, :]  # (N, num_feet, 4)

  pos_b, _ = subtract_frame_transforms(
    anchor_pos_w,
    anchor_quat_w,
    foot_pos_w,
    foot_quat_w,
  )

  return pos_b.reshape(env.num_envs, -1)  # (N, num_feet * 3)
