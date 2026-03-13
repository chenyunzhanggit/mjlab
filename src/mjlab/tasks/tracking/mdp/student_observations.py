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


def ref_hand_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  hand_body_names: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  ),
) -> torch.Tensor:
  """Robot hand end-effector positions expressed in the anchor (torso) body frame.

  Returns the actual robot state (not the reference motion), so it is
  fully available at deployment from forward kinematics / encoders.
  Each hand position is a 3-vector in the torso frame, giving a total
  output of ``len(hand_body_names) * 3`` dimensions.

  Default: left + right ``wrist_yaw_link`` → shape ``(num_envs, 6)``
  """
  command = cast(MultiMotionCommand, env.command_manager.get_term(command_name))

  # Resolve body indices once per call (cheap lookup, no cache needed).
  hand_indexes = torch.tensor(
    command.robot.find_bodies(hand_body_names, preserve_order=True)[0],
    dtype=torch.long,
    device=env.device,
  )

  num_hands = len(hand_body_names)
  # Robot body state in world frame: (N, num_hands, 3/4)
  hand_pos_w = command.robot.data.body_link_pos_w[:, hand_indexes]
  hand_quat_w = command.robot.data.body_link_quat_w[:, hand_indexes]

  # Anchor (torso) pose broadcast to match body batch dim
  anchor_pos_w = command.robot_anchor_pos_w[:, None, :].expand(-1, num_hands, -1)
  anchor_quat_w = command.robot_anchor_quat_w[:, None, :].expand(-1, num_hands, -1)

  # Express hand positions in the anchor frame
  pos_b, _ = subtract_frame_transforms(
    anchor_pos_w,
    anchor_quat_w,
    hand_pos_w,
    hand_quat_w,
  )

  return pos_b.reshape(env.num_envs, -1)  # (N, num_hands * 3)
