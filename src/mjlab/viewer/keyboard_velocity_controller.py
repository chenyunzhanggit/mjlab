"""Keyboard controller for velocity commands during play."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.viewer.native.keys import (
  KEY_A,
  KEY_D,
  KEY_E,
  KEY_Q,
  KEY_S,
  KEY_T,
  KEY_W,
  KEY_X,
)

if TYPE_CHECKING:
  from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand


class KeyboardVelocityController:
  """Override the twist command with keyboard-controlled velocities.

  Controls (toggle with T):
    W / S  — increase / decrease forward velocity (lin_vel_x)
    A / D  — increase / decrease yaw rate (ang_vel_z, left / right)
    Q / E  — increase / decrease lateral velocity (lin_vel_y)
    X      — zero all velocities

  Each key press changes the velocity by ``step_lin`` or ``step_ang``.
  Values are clamped to the command's configured ranges.
  """

  def __init__(
    self,
    twist_cmd: UniformVelocityCommand,
    device: torch.device | str,
    step_lin: float = 0.1,
    step_ang: float = 0.1,
  ):
    self.twist_cmd = twist_cmd
    self.device = device
    self.step_lin = step_lin
    self.step_ang = step_ang
    self.active = False

    cfg_ranges = twist_cmd.cfg.ranges
    self.lim_vx = cfg_ranges.lin_vel_x
    self.lim_vy = cfg_ranges.lin_vel_y
    self.lim_wz = cfg_ranges.ang_vel_z

    self._vel = [0.0, 0.0, 0.0]  # [lin_x, lin_y, ang_z]

  def on_key(self, key: int) -> None:
    if key == KEY_T:
      self.active = not self.active
      state = "ON" if self.active else "OFF"
      print(f"[Keyboard] Velocity control {state}  vel={self._fmt()}")
      if self.active:
        self._apply()
      return

    if not self.active:
      return

    changed = True
    if key == KEY_W:
      self._vel[0] += self.step_lin
    elif key == KEY_S:
      self._vel[0] -= self.step_lin
    elif key == KEY_Q:
      self._vel[1] += self.step_lin
    elif key == KEY_E:
      self._vel[1] -= self.step_lin
    elif key == KEY_A:
      self._vel[2] += self.step_ang
    elif key == KEY_D:
      self._vel[2] -= self.step_ang
    elif key == KEY_X:
      self._vel = [0.0, 0.0, 0.0]
    else:
      changed = False

    if changed:
      self._clamp()
      print(f"[Keyboard] vel={self._fmt()}")
      self._apply()

  def post_step(self) -> None:
    """Call after env.step() to re-apply keyboard command (prevents resampling override)."""
    if self.active:
      self._apply()

  def _apply(self) -> None:
    cmd = self.twist_cmd
    cmd.vel_command_b[:, 0] = self._vel[0]
    cmd.vel_command_b[:, 1] = self._vel[1]
    cmd.vel_command_b[:, 2] = self._vel[2]
    # Prevent the command manager from resampling while keyboard is active.
    cmd.time_left[:] = 1e9
    # Disable standing / heading overrides.
    cmd.is_standing_env[:] = False
    cmd.is_heading_env[:] = False

  def _clamp(self) -> None:
    self._vel[0] = max(self.lim_vx[0], min(self.lim_vx[1], self._vel[0]))
    self._vel[1] = max(self.lim_vy[0], min(self.lim_vy[1], self._vel[1]))
    self._vel[2] = max(self.lim_wz[0], min(self.lim_wz[1], self._vel[2]))

  def _fmt(self) -> str:
    return f"vx={self._vel[0]:+.2f}  vy={self._vel[1]:+.2f}  wz={self._vel[2]:+.2f}"
