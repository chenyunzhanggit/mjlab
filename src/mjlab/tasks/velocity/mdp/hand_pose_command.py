"""Random hand pose and base height commands for post-training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _euler_to_rotmat_cols12(
  roll: torch.Tensor,
  pitch: torch.Tensor,
  yaw: torch.Tensor,
) -> torch.Tensor:
  """Return first two columns of Rz(yaw)@Ry(pitch)@Rx(roll), flattened.

  Input shapes: ``(N,)``.  Output shape: ``(N, 6)`` — [col0 (3D), col1 (3D)].
  """
  cr, sr = torch.cos(roll), torch.sin(roll)
  cp, sp = torch.cos(pitch), torch.sin(pitch)
  cy, sy = torch.cos(yaw), torch.sin(yaw)
  # Column 0 of R
  c0 = torch.stack([cy * cp, sy * cp, -sp], dim=-1)  # (N, 3)
  # Column 1 of R
  c1 = torch.stack(
    [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr], dim=-1
  )  # (N, 3)
  return torch.cat([c0, c1], dim=-1)  # (N, 6)


class RandomHandPoseCommand(CommandTerm):
  """Samples random target hand poses (position + orientation) in the torso frame.

  Stores:
    - ``target_hand_pos_b``: shape ``(num_envs, 6)``  — [left_xyz, right_xyz]
    - ``target_hand_ori_b``: shape ``(num_envs, 12)`` — [left_6D, right_6D]
      orientation encoded as first two columns of rotation matrix
  """

  cfg: RandomHandPoseCommandCfg

  def __init__(self, cfg: RandomHandPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    from mjlab.entity import Entity

    self.robot: Entity = env.scene[cfg.entity_name]
    self.anchor_body_idx: int = self.robot.body_names.index(cfg.anchor_body_name)

    self.target_hand_pos_b = torch.zeros(self.num_envs, 6, device=self.device)
    self.target_hand_ori_b = torch.zeros(self.num_envs, 12, device=self.device)
    # Initialise with a first sample.
    self._resample_command(torch.arange(self.num_envs, device=self.device))

  @property
  def command(self) -> torch.Tensor:
    return self.target_hand_pos_b

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    lr = self.cfg.ranges
    # Left wrist position (indices 0-2)
    self.target_hand_pos_b[env_ids, 0] = r.uniform_(*lr.left_x)
    self.target_hand_pos_b[env_ids, 1] = r.uniform_(*lr.left_y)
    self.target_hand_pos_b[env_ids, 2] = r.uniform_(*lr.left_z)
    # Right wrist position (indices 3-5)
    self.target_hand_pos_b[env_ids, 3] = r.uniform_(*lr.right_x)
    self.target_hand_pos_b[env_ids, 4] = r.uniform_(*lr.right_y)
    self.target_hand_pos_b[env_ids, 5] = r.uniform_(*lr.right_z)
    # Left wrist orientation (indices 0-5)
    ori = self.cfg.ori_ranges
    left_cols = _euler_to_rotmat_cols12(
      r.uniform_(*ori.left_roll),
      r.uniform_(*ori.left_pitch),
      r.uniform_(*ori.left_yaw),
    )
    self.target_hand_ori_b[env_ids, :6] = left_cols
    # Right wrist orientation (indices 6-11)
    right_cols = _euler_to_rotmat_cols12(
      r.uniform_(*ori.right_roll),
      r.uniform_(*ori.right_pitch),
      r.uniform_(*ori.right_yaw),
    )
    self.target_hand_ori_b[env_ids, 6:] = right_cols

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw target hand positions as spheres and orientations as coordinate frames."""

    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    anchor_pos_w = self.robot.data.body_link_pos_w[:, self.anchor_body_idx, :]
    anchor_quat_w = self.robot.data.body_link_quat_w[:, self.anchor_body_idx, :]

    left_b = self.target_hand_pos_b[:, :3]
    right_b = self.target_hand_pos_b[:, 3:]

    left_w = anchor_pos_w + quat_apply(anchor_quat_w, left_b)
    right_w = anchor_pos_w + quat_apply(anchor_quat_w, right_b)

    # Recover full 3x3 rotation matrices from stored 6D (first two cols).
    # col2 = normalize(cross(col0, col1)), col1_orth = cross(col2, col0)
    def _rotmat_from_6d(cols6: torch.Tensor) -> torch.Tensor:
      """cols6: (N, 6) → (N, 3, 3)"""
      c0 = torch.nn.functional.normalize(cols6[:, :3], dim=-1)
      c1_raw = cols6[:, 3:]
      c2 = torch.nn.functional.normalize(torch.cross(c0, c1_raw, dim=-1), dim=-1)
      c1 = torch.cross(c2, c0, dim=-1)
      return torch.stack([c0, c1, c2], dim=-1)  # (N, 3, 3)

    # Rotate the target orientations into world frame.
    # R_w = R_anchor_w @ R_b  (both as 3x3)
    from mjlab.utils.lab_api.math import matrix_from_quat

    anchor_rot_w = matrix_from_quat(anchor_quat_w)  # (N, 3, 3)
    left_rot_b = _rotmat_from_6d(self.target_hand_ori_b[:, :6])  # (N, 3, 3)
    right_rot_b = _rotmat_from_6d(self.target_hand_ori_b[:, 6:])  # (N, 3, 3)
    left_rot_w = torch.bmm(anchor_rot_w, left_rot_b)  # (N, 3, 3)
    right_rot_w = torch.bmm(anchor_rot_w, right_rot_b)  # (N, 3, 3)

    left_w_np = left_w.cpu().numpy()
    right_w_np = right_w.cpu().numpy()
    left_rot_w_np = left_rot_w.cpu().numpy()
    right_rot_w_np = right_rot_w.cpu().numpy()

    for i in env_indices:
      visualizer.add_sphere(
        center=left_w_np[i],
        radius=0.03,
        color=(0.2, 0.4, 1.0, 0.8),  # blue — left hand
        label="left_target",
      )
      visualizer.add_frame(
        position=left_w_np[i],
        rotation_matrix=left_rot_w_np[i],
        scale=0.1,
        label="left_ori",
      )
      visualizer.add_sphere(
        center=right_w_np[i],
        radius=0.03,
        color=(1.0, 0.3, 0.2, 0.8),  # red — right hand
        label="right_target",
      )
      visualizer.add_frame(
        position=right_w_np[i],
        rotation_matrix=right_rot_w_np[i],
        scale=0.1,
        label="right_ori",
      )


@dataclass(kw_only=True)
class RandomHandPoseCommandCfg(CommandTermCfg):
  """Configuration for :class:`RandomHandPoseCommand`."""

  entity_name: str = "robot"
  """Scene entity name for the robot."""

  anchor_body_name: str = "torso_link"
  """Body used as the reference frame for hand positions."""

  @dataclass
  class Ranges:
    """Clamp bounds for delta-sampled hand positions in the torso frame."""

    left_x: tuple[float, float] = (0.0, 0.28)
    """Left wrist x (forward)."""
    left_y: tuple[float, float] = (0.0, 0.40)
    """Left wrist y (outward, positive for left arm)."""
    left_z: tuple[float, float] = (0.0, 0.45)
    """Left wrist z (vertical, relative to torso)."""

    right_x: tuple[float, float] = (0.0, 0.28)
    """Right wrist x (forward)."""
    right_y: tuple[float, float] = (-0.40, 0.0)
    """Right wrist y (outward, negative for right arm)."""
    right_z: tuple[float, float] = (0.0, 0.45)
    """Right wrist z (vertical, relative to torso)."""

  @dataclass
  class OriRanges:
    """Euler-angle sampling ranges (radians) for hand orientation in torso frame."""

    left_roll: tuple[float, float] = (-1.2, 1.2)
    left_pitch: tuple[float, float] = (-1.2, 1.2)
    left_yaw: tuple[float, float] = (-1.2, 1.2)

    right_roll: tuple[float, float] = (-1.2, 1.2)
    right_pitch: tuple[float, float] = (-1.2, 1.2)
    right_yaw: tuple[float, float] = (-1.2, 1.2)

  ranges: Ranges = field(default_factory=Ranges)
  ori_ranges: OriRanges = field(default_factory=OriRanges)

  def build(self, env: ManagerBasedRlEnv) -> RandomHandPoseCommand:
    return RandomHandPoseCommand(self, env)


# ── Base height command ────────────────────────────────────────────────────


class RandomBaseHeightCommand(CommandTerm):
  """Samples a random target torso height each episode.

  Stores:
    - ``target_height``: shape ``(num_envs, 1)`` — target torso height (metres)
  """

  cfg: RandomBaseHeightCommandCfg

  def __init__(self, cfg: RandomBaseHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.target_height = torch.full(
      (self.num_envs, 1), cfg.nominal_height, device=self.device
    )
    self._resample_command(torch.arange(self.num_envs, device=self.device))

  @property
  def command(self) -> torch.Tensor:
    return self.target_height

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.randomize:
      r = torch.empty(len(env_ids), device=self.device)
      self.target_height[env_ids, 0] = r.uniform_(*self.cfg.height_range)

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class RandomBaseHeightCommandCfg(CommandTermCfg):
  """Configuration for :class:`RandomBaseHeightCommand`."""

  nominal_height: float = 0.78
  """Fixed target torso height (metres). Used when ``randomize=False``."""

  randomize: bool = False
  """If True, sample height uniformly from ``height_range`` each episode."""

  height_range: tuple[float, float] = (0.70, 0.85)
  """Height sampling range used when ``randomize=True``."""

  def build(self, env: ManagerBasedRlEnv) -> RandomBaseHeightCommand:
    return RandomBaseHeightCommand(self, env)
