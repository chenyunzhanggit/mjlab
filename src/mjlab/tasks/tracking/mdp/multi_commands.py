from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field, MISSING
from typing import TYPE_CHECKING, Literal, Optional, Sequence

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MultiMotionLoader:
    def __init__(self, motion_files: list[str], body_indexes: torch.Tensor, device: str = "cpu"):
        assert len(motion_files) > 0, "motion_files cannot be empty"
        self.num_files = len(motion_files)
        self._body_indexes = body_indexes
        self.device = device

        # 存储每个motion的数据为list，不进行填充
        self.joint_pos_list = []
        self.joint_vel_list = []
        self._body_pos_w_list = []
        self._body_quat_w_list = []
        self._body_lin_vel_w_list = []
        self._body_ang_vel_w_list = []
        self.fps_list = []
        self.file_lengths = []

        self.isaaclab_joint_names = [
            'left_hip_pitch_joint', 
            'right_hip_pitch_joint',
            'waist_yaw_joint',
            'left_hip_roll_joint',
            'right_hip_roll_joint',
            'waist_roll_joint',
            'left_hip_yaw_joint',
            'right_hip_yaw_joint',
            'waist_pitch_joint',
            'left_knee_joint',
            'right_knee_joint',
            'left_shoulder_pitch_joint',
            'right_shoulder_pitch_joint',
            'left_ankle_pitch_joint',
            'right_ankle_pitch_joint',
            'left_shoulder_roll_joint',
            'right_shoulder_roll_joint',
            'left_ankle_roll_joint',
            'right_ankle_roll_joint',
            'left_shoulder_yaw_joint',
            'right_shoulder_yaw_joint',
            'left_elbow_joint',
            'right_elbow_joint',
            'left_wrist_roll_joint',
            'right_wrist_roll_joint',
            'left_wrist_pitch_joint',
            'right_wrist_pitch_joint',
            'left_wrist_yaw_joint',
            'right_wrist_yaw_joint'
        ]

        self.mujoco_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        self.isaaclab_body_names = ['pelvis', 'left_hip_pitch_link', 'right_hip_pitch_link', 'waist_yaw_link', 'left_hip_roll_link', 'right_hip_roll_link', 'waist_roll_link', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_knee_link', 'right_knee_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_ankle_pitch_link', 'right_ankle_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_elbow_link', 'right_elbow_link', 'left_wrist_roll_link', 'right_wrist_roll_link', 'left_wrist_pitch_link', 'right_wrist_pitch_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link']
        self.mujoco_body_names = ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link']

        self.isaaclab_to_mujoco_joint_reindex = [self.isaaclab_joint_names.index(name) for name in self.mujoco_joint_names]
        self.mujoco_to_isaaclab_joint_reindex = [self.mujoco_joint_names.index(name) for name in self.isaaclab_joint_names]
        self.isaaclab_to_mujoco_body_reindex = [self.isaaclab_body_names.index(name) for name in self.mujoco_body_names]
        self.mujoco_to_isaaclab_body_reindex = [self.mujoco_body_names.index(name) for name in self.isaaclab_body_names]
        


        for motion_file in motion_files:
            assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
            data = np.load(motion_file)

            self.fps_list.append(data["fps"])

            jp = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)[:, self.isaaclab_to_mujoco_joint_reindex]
            jv = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)[:, self.isaaclab_to_mujoco_joint_reindex]
            bp = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)[:, self.isaaclab_to_mujoco_body_reindex, :]
            bq = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)[:, self.isaaclab_to_mujoco_body_reindex, :]
            blv = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)[:, self.isaaclab_to_mujoco_body_reindex, :]
            bav = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)[:, self.isaaclab_to_mujoco_body_reindex, :]
       
            self.joint_pos_list.append(jp)
            self.joint_vel_list.append(jv)
            self._body_pos_w_list.append(bp)
            self._body_quat_w_list.append(bq)
            self._body_lin_vel_w_list.append(blv)
            self._body_ang_vel_w_list.append(bav)
            self.file_lengths.append(jp.shape[0])

        self.file_lengths = torch.tensor(self.file_lengths, dtype=torch.long, device=self.device)
        self.fps = self.fps_list[0]  # 可以根据需求调整

    def get_motion_data_batch(
        self, motion_idx: int, time_steps_start: torch.Tensor, time_steps_end: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        time_steps_tensor = torch.arange(
            time_steps_start.item(), time_steps_end.item(), device=self.device, dtype=torch.long
        )
        time_steps_tensor = torch.clamp(
            time_steps_tensor, torch.tensor(0, device=self.device), self.file_lengths[motion_idx] - 1
        )

        return {
            "joint_pos": self.joint_pos_list[motion_idx][time_steps_tensor],
            "joint_vel": self.joint_vel_list[motion_idx][time_steps_tensor],
            "body_pos_w": self._body_pos_w_list[motion_idx][time_steps_tensor][:, self._body_indexes],
            "body_quat_w": self._body_quat_w_list[motion_idx][time_steps_tensor][:, self._body_indexes],
            "body_lin_vel_w": self._body_lin_vel_w_list[motion_idx][time_steps_tensor][:, self._body_indexes],
            "body_ang_vel_w": self._body_ang_vel_w_list[motion_idx][time_steps_tensor][:, self._body_indexes],
        }


class MultiMotionCommand(CommandTerm):
    cfg: "MultiMotionCommandCfg"
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: "MultiMotionCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        # Validate motion_files
        if not self.cfg.motion_files or len(self.cfg.motion_files) == 0:
            raise ValueError(
                "motion_files cannot be empty. Provide either:\n"
                "  - motion_files: list of .npz file paths\n"
                "  - motion_path: path to directory containing motion.npz files"
            )

        self.motion = MultiMotionLoader(
            self.cfg.motion_files, self.body_indexes, device=self.device
        )

        # Calculate buffer length based on max episode length and motion length
        max_episode_length = int(self._env.max_episode_length_s / self._env.step_dt) if self._env.max_episode_length_s > 0 else self.motion.file_lengths.max().item()
        self.buffer_length: int = int(
            min(max_episode_length, self.motion.file_lengths.max().item())
        ) + self.cfg.future_steps + self.cfg.history_steps
        # 初始化状态变量
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.buffer_start_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 初始化buffer，存储轨迹数据
        self._init_buffers()

        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        # 计算能分成多少个bin，step_dt是一个仿真步长的时间，1 / step_dt是一秒内有多少仿真步长时间
        # 也就是说每个bin对应1秒，最后bin_count就是可以分成的bin的个数
        max_motion_length = self.motion.file_lengths.max().item()
        self.bin_count = int(max_motion_length // (1 / env.step_dt)) + 1
        self.bin_failed_count = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        if self.cfg.if_log_metrics:
            self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

        # 记录躺下环境的 mask 和 env_ids
        self.init_fall_recovery_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.init_fall_recovery_env_ids: torch.Tensor = torch.tensor([], dtype=torch.long, device=self.device)

        # Ghost model created lazily on first visualization
        self._ghost_model: mujoco.MjModel | None = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    def _init_buffers(self):
        """初始化buffer存储轨迹数据"""
        # 获取joint数量
        joint_dim = self.motion.joint_pos_list[0].shape[1]
        body_dim = len(self.cfg.body_names)

        # 初始化buffer，形状为 (num_envs, buffer_length, ...)
        self.joint_pos_buffer = torch.zeros(
            self.num_envs, self.buffer_length, joint_dim, device=self.device
        )
        self.joint_vel_buffer = torch.zeros(
            self.num_envs, self.buffer_length, joint_dim, device=self.device
        )
        self.body_pos_w_buffer = torch.zeros(
            self.num_envs, self.buffer_length, body_dim, 3, device=self.device
        )
        self.body_quat_w_buffer = torch.zeros(
            self.num_envs, self.buffer_length, body_dim, 4, device=self.device
        )
        self.body_lin_vel_w_buffer = torch.zeros(
            self.num_envs, self.buffer_length, body_dim, 3, device=self.device
        )
        self.body_ang_vel_w_buffer = torch.zeros(
            self.num_envs, self.buffer_length, body_dim, 3, device=self.device
        )

        # 初始化quaternion为[1,0,0,0]
        self.body_quat_w_buffer[:, :, :, 0] = 1.0

    def _update_buffers(self, env_ids: Optional[torch.Tensor] = None):
        """更新buffer，从motion数据中填充buffer"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if len(env_ids) == 0:
            return

        for env_id in env_ids:
            motion_data = self.motion.get_motion_data_batch(
                int(self.motion_idx[env_id].item()),
                self.buffer_start_time[env_id],
                self.buffer_start_time[env_id] + self.buffer_length,
            )
            self.joint_pos_buffer[env_id] = motion_data["joint_pos"]
            self.joint_vel_buffer[env_id] = motion_data["joint_vel"]
            self.body_pos_w_buffer[env_id] = motion_data["body_pos_w"]
            self.body_quat_w_buffer[env_id] = motion_data["body_quat_w"]
            self.body_lin_vel_w_buffer[env_id] = motion_data["body_lin_vel_w"]
            self.body_ang_vel_w_buffer[env_id] = motion_data["body_ang_vel_w"]

    @property
    def command(self) -> torch.Tensor:
        cmd = torch.cat([self.motion_joint_pos, self.motion_joint_vel], dim=1)
        return cmd

    @property
    def command_joint_pos(self) -> torch.Tensor:
        return self.motion_joint_pos

    @property
    def command_current_joint_pos(self) -> torch.Tensor:
        return self.current_motion_joint_pos

    @property
    def joint_pos(self) -> torch.Tensor:
        buffer_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        return self.joint_pos_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices
        ]

    @property
    def joint_vel(self) -> torch.Tensor:
        buffer_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        return self.joint_vel_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices
        ]

    @property
    def body_pos_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        return (
            self.body_pos_w_buffer[
                torch.arange(self.num_envs, device=self.device), buffer_indices
            ]
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        return self.body_quat_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices
        ]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        return self.body_lin_vel_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices
        ]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        return self.body_ang_vel_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices
        ]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        return (
            self.body_pos_w_buffer[
                torch.arange(self.num_envs, device=self.device),
                buffer_indices,
                self.motion_anchor_body_index,
            ]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """Anchor quaternions at current step."""
        buffer_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        return self.body_quat_w_buffer[
            torch.arange(self.num_envs, device=self.device),
            buffer_indices,
            self.motion_anchor_body_index,
        ]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """Anchor linear velocities with history and future steps.
        
        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        """
        current_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        
        parts = []
        
        # Order should be [past, ..., recent_past] (from oldest to most recent)
        if self.cfg.history_steps > 0:
            # Get history indices excluding current step: [current-1, current-2, ..., current-history_steps]
            history_indices = current_indices[:, None] - torch.arange(
                1, self.cfg.history_steps + 1, device=self.device
            )[None, :]
            history_indices = torch.clamp(history_indices, 0, self.buffer_length - 1)
            # Reverse to get [oldest, ..., most_recent] order
            history_indices = history_indices.flip(dims=[1])
            history_data = self.body_lin_vel_w_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                history_indices,
                self.motion_anchor_body_index,
            ]
            parts.append(history_data)
        
        # Add current step
        current_data = self.body_lin_vel_w_buffer[
            torch.arange(self.num_envs, device=self.device),
            current_indices,
            self.motion_anchor_body_index,
        ].unsqueeze(1)
        parts.append(current_data)
        
        # Add future steps (forwards from current, excluding current since it's already added)
        if self.cfg.future_steps > 1:
            future_indices = current_indices[:, None] + torch.arange(
                1, self.cfg.future_steps, device=self.device
            )[None, :]
            future_indices = torch.clamp(future_indices, 0, self.buffer_length - 1)
            future_data = self.body_lin_vel_w_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                future_indices,
                self.motion_anchor_body_index,
            ]
            parts.append(future_data)
        
        # Concatenate all parts along the time dimension
        if len(parts) > 1:
            return torch.cat(parts, dim=1).view(self.num_envs, -1)
        else:
            return parts[0].view(self.num_envs, -1)

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """Anchor angular velocities with history and future steps.
        
        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        """
        current_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        
        parts = []
        
        # Order should be [past, ..., recent_past] (from oldest to most recent)
        if self.cfg.history_steps > 0:
            # Get history indices excluding current step: [current-1, current-2, ..., current-history_steps]
            history_indices = current_indices[:, None] - torch.arange(
                1, self.cfg.history_steps + 1, device=self.device
            )[None, :]
            history_indices = torch.clamp(history_indices, 0, self.buffer_length - 1)
            history_indices = history_indices.flip(dims=[1])
            history_data = self.body_ang_vel_w_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                history_indices,
                self.motion_anchor_body_index,
            ]
            parts.append(history_data)
        
        # Add current step
        current_data = self.body_ang_vel_w_buffer[
            torch.arange(self.num_envs, device=self.device),
            current_indices,
            self.motion_anchor_body_index,
        ].unsqueeze(1)
        parts.append(current_data)
        
        # Add future steps (forwards from current, excluding current since it's already added)
        if self.cfg.future_steps > 1:
            future_indices = current_indices[:, None] + torch.arange(
                1, self.cfg.future_steps, device=self.device
            )[None, :]
            future_indices = torch.clamp(future_indices, 0, self.buffer_length - 1)
            future_data = self.body_ang_vel_w_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                future_indices,
                self.motion_anchor_body_index,
            ]
            parts.append(future_data)
        
        # Concatenate all parts along the time dimension
        if len(parts) > 1:
            return torch.cat(parts, dim=1).view(self.num_envs, -1)
        else:
            return parts[0].view(self.num_envs, -1)

    @property
    def anchor_projected_gravity(self) -> torch.Tensor:
        """Anchor projected gravity with history and future steps.
        
        Converts anchor quaternions to projected gravity vectors using the formula:
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        
        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        Shape: (num_envs, num_steps * 3) where num_steps = history_steps + 1 + (future_steps - 1)
        """
        current_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        
        parts = []
        
        # Order should be [past, ..., recent_past] (from oldest to most recent)
        if self.cfg.history_steps > 0:
            # Get history indices excluding current step: [current-1, current-2, ..., current-history_steps]
            history_indices = current_indices[:, None] - torch.arange(
                1, self.cfg.history_steps + 1, device=self.device
            )[None, :]
            history_indices = torch.clamp(history_indices, 0, self.buffer_length - 1)
            history_indices = history_indices.flip(dims=[1])
            history_quat = self.body_quat_w_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                history_indices,
                self.motion_anchor_body_index,
            ]
            parts.append(history_quat)
        
        # Add current step
        current_quat = self.body_quat_w_buffer[
            torch.arange(self.num_envs, device=self.device),
            current_indices,
            self.motion_anchor_body_index,
        ].unsqueeze(1)
        parts.append(current_quat)
        
        # Add future steps (forwards from current, excluding current since it's already added)
        if self.cfg.future_steps > 1:
            future_indices = current_indices[:, None] + torch.arange(
                1, self.cfg.future_steps, device=self.device
            )[None, :]
            future_indices = torch.clamp(future_indices, 0, self.buffer_length - 1)
            future_quat = self.body_quat_w_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                future_indices,
                self.motion_anchor_body_index,
            ]
            parts.append(future_quat)
        
        # Concatenate all quaternion parts along the time dimension
        if len(parts) > 1:
            anchor_quat = torch.cat(parts, dim=1)  # Shape: (num_envs, num_steps, 4)
        else:
            anchor_quat = parts[0]  # Shape: (num_envs, 1, 4)
        
        # Extract quaternion components: (w, x, y, z) format
        qw = anchor_quat[..., 0]  # (num_envs, num_steps)
        qx = anchor_quat[..., 1]  # (num_envs, num_steps)
        qy = anchor_quat[..., 2]  # (num_envs, num_steps)
        qz = anchor_quat[..., 3]  # (num_envs, num_steps)
        
        # Compute projected gravity for each step
        gravity_x = 2 * (-qz * qx + qw * qy)
        gravity_y = -2 * (qz * qy + qw * qx)
        gravity_z = 1 - 2 * (qw * qw + qz * qz)
        
        # Stack to (num_envs, num_steps, 3)
        projected_gravity = torch.stack([gravity_x, gravity_y, gravity_z], dim=-1)
        
        # Reshape to (num_envs, num_steps * 3)
        return projected_gravity.view(self.num_envs, -1)

    # Motion reference properties with history and future steps
    @property
    def motion_joint_pos(self) -> torch.Tensor:
        """Joint positions reference with history and future steps.
        
        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        """
        current_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        
        parts = []
        
        # Order should be [past, ..., recent_past] (from oldest to most recent)
        if self.cfg.history_steps > 0:
            # Get history indices excluding current step: [current-1, current-2, ..., current-history_steps]
            history_indices = current_indices[:, None] - torch.arange(
                1, self.cfg.history_steps + 1, device=self.device
            )[None, :]
            history_indices = torch.clamp(history_indices, 0, self.buffer_length - 1)
            history_indices = history_indices.flip(dims=[1])
            history_data = self.joint_pos_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None], history_indices
            ]
            parts.append(history_data)
        
        # Add current step
        current_data = self.joint_pos_buffer[
            torch.arange(self.num_envs, device=self.device), current_indices
        ].unsqueeze(1)
        parts.append(current_data)
        
        # Add future steps (forwards from current, excluding current since it's already added)
        if self.cfg.future_steps > 1:
            future_indices = current_indices[:, None] + torch.arange(
                1, self.cfg.future_steps, device=self.device
            )[None, :]
            future_indices = torch.clamp(future_indices, 0, self.buffer_length - 1)
            future_data = self.joint_pos_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None], future_indices
            ]
            parts.append(future_data)

        # Concatenate all parts along the time dimension
        
        if len(parts) > 1:
            return torch.cat(parts, dim=1).view(self.num_envs, -1)
        else:
            return parts[0].view(self.num_envs, -1)

    @property
    def current_motion_joint_pos(self) -> torch.Tensor:
        """Joint positions reference at current step only."""
        buffer_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        return self.joint_pos_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices
        ]

    @property
    def motion_joint_vel(self) -> torch.Tensor:
        """Joint velocities reference with history and future steps.
        
        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        """
        current_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        
        parts = []
        
        # Order should be [past, ..., recent_past] (from oldest to most recent)
        if self.cfg.history_steps > 0:
            # Get history indices excluding current step: [current-1, current-2, ..., current-history_steps]
            history_indices = current_indices[:, None] - torch.arange(
                1, self.cfg.history_steps + 1, device=self.device
            )[None, :]
            history_indices = torch.clamp(history_indices, 0, self.buffer_length - 1)
            # Reverse to get [oldest, ..., most_recent] order
            history_indices = history_indices.flip(dims=[1])
            history_data = self.joint_vel_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None], history_indices
            ]
            parts.append(history_data)
        
        # Add current step
        current_data = self.joint_vel_buffer[
            torch.arange(self.num_envs, device=self.device), current_indices
        ].unsqueeze(1)
        parts.append(current_data)
        
        # Add future steps (forwards from current)
        if self.cfg.future_steps > 1:
            future_indices = current_indices[:, None] + torch.arange(
                1, self.cfg.future_steps, device=self.device
            )[None, :]
            future_indices = torch.clamp(future_indices, 0, self.buffer_length - 1)
            future_data = self.joint_vel_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None], future_indices
            ]
            parts.append(future_data)

        # Concatenate all parts along the time dimension
        if len(parts) > 1:
            return torch.cat(parts, dim=1).view(self.num_envs, -1)
        else:
            return parts[0].view(self.num_envs, -1)

    @property
    def motion_anchor_pos(self) -> torch.Tensor:
        """Anchor positions reference with history and future steps.
        
        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        """
        current_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        
        parts = []
        
        # Order should be [past, ..., recent_past] (from oldest to most recent)
        if self.cfg.history_steps > 0:
            # Get history indices excluding current step: [current-1, current-2, ..., current-history_steps]
            history_indices = current_indices[:, None] - torch.arange(
                1, self.cfg.history_steps + 1, device=self.device
            )[None, :]
            history_indices = torch.clamp(history_indices, 0, self.buffer_length - 1)
            history_indices = history_indices.flip(dims=[1])
            history_pos = self.body_pos_w_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                history_indices,
                self.motion_anchor_body_index,
            ]
            parts.append(history_pos)
        
        # Add current step
        current_pos = self.body_pos_w_buffer[
            torch.arange(self.num_envs, device=self.device),
            current_indices,
            self.motion_anchor_body_index,
        ].unsqueeze(1)
        parts.append(current_pos)
        
        # Add future steps (forwards from current)
        if self.cfg.future_steps > 1:
            future_indices = current_indices[:, None] + torch.arange(
                1, self.cfg.future_steps, device=self.device
            )[None, :]
            future_indices = torch.clamp(future_indices, 0, self.buffer_length - 1)
            future_pos = self.body_pos_w_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                future_indices,
                self.motion_anchor_body_index,
            ]
            parts.append(future_pos)
        
        # Concatenate all parts and add env_origins
        if len(parts) > 1:
            combined = torch.cat(parts, dim=1)
        else:
            combined = parts[0]
 
        return (combined + self._env.scene.env_origins[:, None, :]).view(
            self.num_envs, -1
        )

    @property
    def motion_anchor_quat(self) -> torch.Tensor:
        """Anchor quaternions reference with history and future steps.
        
        Returns concatenated [history_steps, current, future_steps] if both are enabled,
        or just the enabled steps. Order: [past, current, future].
        """
        current_indices = torch.clamp(
            self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
        )
        
        parts = []
        
        # Order should be [past, ..., recent_past] (from oldest to most recent)
        if self.cfg.history_steps > 0:
            # Get history indices excluding current step: [current-1, current-2, ..., current-history_steps]
            history_indices = current_indices[:, None] - torch.arange(
                1, self.cfg.history_steps + 1, device=self.device
            )[None, :]
            history_indices = torch.clamp(history_indices, 0, self.buffer_length - 1)
            history_indices = history_indices.flip(dims=[1])
            history_quat = self.body_quat_w_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                history_indices,
                self.motion_anchor_body_index,
            ]
            parts.append(history_quat)
        
        # Add current step
        current_quat = self.body_quat_w_buffer[
            torch.arange(self.num_envs, device=self.device),
            current_indices,
            self.motion_anchor_body_index,
        ].unsqueeze(1)
        parts.append(current_quat)
        
        # Add future steps (forwards from current)
        if self.cfg.future_steps > 1:
            future_indices = current_indices[:, None] + torch.arange(
                1, self.cfg.future_steps, device=self.device
            )[None, :]
            future_indices = torch.clamp(future_indices, 0, self.buffer_length - 1)
            future_quat = self.body_quat_w_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                future_indices,
                self.motion_anchor_body_index,
            ]
            parts.append(future_quat)
 
        # Concatenate all parts along the time dimension
        if len(parts) > 1:
            return torch.cat(parts, dim=1).view(self.num_envs, -1)
        else:
            return parts[0].view(self.num_envs, -1)

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        if not self.cfg.if_log_metrics:
            return

        # Extract current step data from multi-step properties
        # anchor_lin_vel_w and anchor_ang_vel_w contain [history_steps, current, future_steps]
        # Current step is at index: history_steps
        # Calculate total number of steps: history_steps + 1 (current) + (future_steps - 1)
        num_steps_total = self.cfg.history_steps + 1 + max(0, self.cfg.future_steps - 1)
        current_step_idx = self.cfg.history_steps
        
        # For anchor_lin_vel_w and anchor_ang_vel_w, extract current step
        # Reshape from (num_envs, num_steps * 3) to (num_envs, num_steps, 3) and extract current step
        if num_steps_total > 1:
            anchor_lin_vel_current = self.anchor_lin_vel_w.view(self.num_envs, num_steps_total, 3)[:, current_step_idx, :]
            anchor_ang_vel_current = self.anchor_ang_vel_w.view(self.num_envs, num_steps_total, 3)[:, current_step_idx, :]
        else:
            # No history/future, use directly (shape is already (num_envs, 3))
            anchor_lin_vel_current = self.anchor_lin_vel_w
            anchor_ang_vel_current = self.anchor_ang_vel_w

        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            anchor_lin_vel_current - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            anchor_ang_vel_current - self.robot_anchor_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
        )

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            # 每个env当前的time_steps对应的bin的索引
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count)
                // max(self.motion.file_lengths.max().item(), 1),
                0,
                self.bin_count - 1,
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            # 统计每个bin的失败次数
            self._current_bin_failed[:] = torch.bincount(
                fail_bins, minlength=self.bin_count
            )

        # Sample
        # 添加一点均匀分布的概率，防止有些动作采样概率为0
        sampling_probabilities = (
            self.bin_failed_count
            + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        )
        # 填充只在最后一个维度（length） 上进行
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )  # replicate 保证填充的值不改变统计趋势（用最后一个 bin 的失败概率）
        # 因为后面邻居的失败可能跟前面有关，且越近相关程度越高，所以卷积核采用的是指数形式的，会考虑未来的失败
        sampling_probabilities = torch.nn.functional.conv1d(
            sampling_probabilities, self.kernel.view(1, 1, -1)
        ).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        )

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.file_lengths.max().item() - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _uniform_sampling(self, env_ids: torch.Tensor):
        self.time_steps[env_ids] = (
                sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
                * self.motion_length[env_ids]
        ).long()
        self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
        self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
        self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.

    def _resample_command(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        # get motion index
        self.motion_idx[env_ids] = (
            sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
            * self.motion.num_files
        ).long()
        self.motion_length[env_ids] = self.motion.file_lengths[self.motion_idx[env_ids]]
    
        if self.cfg.sampling_mode == "start":
            self.time_steps[env_ids] = 0
            print(
                " ************** [FOR DEBUG] WARNING: All envs time steps is set to start initialization ! ************** "
            )

        elif self.cfg.sampling_mode == "uniform":
            self._uniform_sampling(env_ids)
        else:
            assert self.cfg.sampling_mode == "adaptive"
            self._adaptive_sampling(env_ids)

        self.buffer_start_time[env_ids] = self.time_steps[env_ids].clone()

        # 填充buffer
        self._update_buffers(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()
        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(
            lower=self.cfg.joint_position_range[0],
            upper=self.cfg.joint_position_range[1],
            size=joint_pos.shape,
            device=joint_pos.device,  # type: ignore
        )
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids],
            soft_joint_pos_limits[:, :, 0],
            soft_joint_pos_limits[:, :, 1],
        )
        
        # 随机选择指定比例的环境设置为躺下状态
        num_envs_to_reset = len(env_ids)
        num_fall_recovery = int(num_envs_to_reset * self.cfg.fall_recovery_ratio)  
        
        # 先清除当前重置环境的躺下标记（因为要重新分配）
        self.init_fall_recovery_mask[env_ids] = False
        
        # 从 env_ids 中随机选择要躺下的环境
        if num_fall_recovery < num_envs_to_reset:
            # 随机打乱 env_ids 并选择前 num_fall_recovery 个
            perm = torch.randperm(num_envs_to_reset, device=self.device)
            fall_recovery_local_indices = perm[:num_fall_recovery]
            fall_recovery_env_ids = env_ids[fall_recovery_local_indices]
        else:
            # 如果比例超过100%，则全部设置为躺下
            fall_recovery_env_ids = env_ids
        
        # 更新躺下环境的 mask
        self.init_fall_recovery_mask[fall_recovery_env_ids] = True
        # 记录躺下环境的 env_ids（只记录当前重置的躺下环境）
        self.init_fall_recovery_env_ids = fall_recovery_env_ids.clone()
        # 对于躺下的环境，应用特殊的初始化配置
        if len(fall_recovery_env_ids) > 0:
            # 从配置中获取 RPY 范围
            roll_range = self.cfg.fall_recovery_pose_range.get("roll", (0.0, 0.0))
            pitch_range = self.cfg.fall_recovery_pose_range.get("pitch", (math.pi / 2.0, math.pi / 2.0))
            yaw_range = self.cfg.fall_recovery_pose_range.get("yaw", (0.0, 0.0)) 

            
            
            # 采样 roll pitch yaw
            fall_recovery_roll = sample_uniform(
                roll_range[0], roll_range[1], (len(fall_recovery_env_ids),), device=self.device
            )
            fall_recovery_pitch = sample_uniform(
                pitch_range[0], pitch_range[1], (len(fall_recovery_env_ids),), device=self.device
            )
            fall_recovery_yaw = sample_uniform(
                yaw_range[0], yaw_range[1], (len(fall_recovery_env_ids),), device=self.device
            )

            # 对 15% 的环境随机乘以 ±1
            flip_num_envs = len(fall_recovery_env_ids)
            if flip_num_envs > 0:
                # 随机选择要翻转的环境索引
                flip_indices = torch.randperm(flip_num_envs, device=self.device)[:flip_num_envs]
                # 随机生成 ±1 的符号
                flip_signs = torch.randint(0, 2, (flip_num_envs,), device=self.device) * 2 - 1  # 生成 -1 或 1
                fall_recovery_pitch[flip_indices] *= flip_signs
            
            fall_recovery_quat = quat_from_euler_xyz(
                fall_recovery_roll, 
                fall_recovery_pitch, 
                fall_recovery_yaw
            )
            
            # 更新躺下环境的方向
            root_ori[fall_recovery_env_ids] = fall_recovery_quat
            
            # 2. 应用躺下状态的关节角度噪声
            self.robot.data.default_joint_pos
            fall_recovery_joint_pos = joint_pos[fall_recovery_env_ids].clone()
            fall_recovery_joint_pos += sample_uniform(
                lower=self.cfg.fall_recovery_joint_position_range[0],
                upper=self.cfg.fall_recovery_joint_position_range[1],
                size=fall_recovery_joint_pos.shape,
                device=str(fall_recovery_joint_pos.device),
            )
            # 限制在关节软限制内
            soft_joint_pos_limits_recovery = self.robot.data.soft_joint_pos_limits[fall_recovery_env_ids]
            fall_recovery_joint_pos = torch.clip(
                fall_recovery_joint_pos,
                soft_joint_pos_limits_recovery[:, :, 0],
                soft_joint_pos_limits_recovery[:, :, 1],
            )
            joint_pos[fall_recovery_env_ids] = fall_recovery_joint_pos
            
            # 3. 应用躺下状态的关节速度噪声
            fall_recovery_joint_vel = joint_vel[fall_recovery_env_ids].clone()
            fall_recovery_joint_vel += sample_uniform(
                lower=self.cfg.fall_recovery_joint_velocity_range[0],
                upper=self.cfg.fall_recovery_joint_velocity_range[1],
                size=fall_recovery_joint_vel.shape,
                device=str(fall_recovery_joint_vel.device),
            )
            joint_vel[fall_recovery_env_ids] = fall_recovery_joint_vel
            
            # 4. 应用躺下状态的速度范围（如果需要）
            if self.cfg.fall_recovery_velocity_range:
                fall_recovery_velocity_range_list = [
                    self.cfg.fall_recovery_velocity_range.get(key, (0.0, 0.0))
                    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
                ]
                fall_recovery_velocity_ranges = torch.tensor(
                    fall_recovery_velocity_range_list, device=self.device
                )
                fall_recovery_velocity_samples = sample_uniform(
                    fall_recovery_velocity_ranges[:, 0],
                    fall_recovery_velocity_ranges[:, 1],
                    (len(fall_recovery_env_ids), 6),
                    device=self.device,
                )
                root_lin_vel[fall_recovery_env_ids] += fall_recovery_velocity_samples[:, :3]
                root_ang_vel[fall_recovery_env_ids] += fall_recovery_velocity_samples[:, 3:]

            # 设置 fall_recovery 环境的初始位置为 (0, 0, 0.2)，相对于 env_origins
            root_pos[fall_recovery_env_ids] = (
                self._env.scene.env_origins[fall_recovery_env_ids] + 
                torch.tensor([0.0, 0.0, 0.20], device=self.device).unsqueeze(0).repeat(len(fall_recovery_env_ids), 1)
            )

        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
        )
        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos[env_ids],
                    root_ori[env_ids],
                    root_lin_vel[env_ids],
                    root_ang_vel[env_ids],
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )

        self.robot.clear_state(env_ids=env_ids)

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion_length)[0]
        if env_ids.numel() > 0:
            self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

        if self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw ghost robot or frames based on visualization mode."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        if self.cfg.viz.mode == "ghost":
            if self._ghost_model is None:
                self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
                self._ghost_model.geom_rgba[:] = self._ghost_color

            entity: Entity = self._env.scene[self.cfg.entity_name]
            indexing = entity.indexing
            free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
            joint_q_adr = indexing.joint_q_adr.cpu().numpy()

            for batch in env_indices:
                qpos = np.zeros(self._env.sim.mj_model.nq)
                qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
                qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
                qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

                visualizer.add_ghost_mesh(
                    qpos, model=self._ghost_model, label=f"ghost_{batch}"
                )

        elif self.cfg.viz.mode == "frames":
            for batch in env_indices:
                desired_body_pos = self.body_pos_w[batch].cpu().numpy()
                desired_body_quat = self.body_quat_w[batch]
                desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

                current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
                current_body_quat = self.robot_body_quat_w[batch]
                current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

                for i, body_name in enumerate(self.cfg.body_names):
                    visualizer.add_frame(
                        position=desired_body_pos[i],
                        rotation_matrix=desired_body_rotm[i],
                        scale=0.08,
                        label=f"desired_{body_name}_{batch}",
                        axis_colors=_DESIRED_FRAME_COLORS,
                    )
                    visualizer.add_frame(
                        position=current_body_pos[i],
                        rotation_matrix=current_body_rotm[i],
                        scale=0.12,
                        label=f"current_{body_name}_{batch}",
                    )

                desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
                desired_anchor_quat = self.anchor_quat_w[batch]
                desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
                visualizer.add_frame(
                    position=desired_anchor_pos,
                    rotation_matrix=desired_rotation_matrix,
                    scale=0.1,
                    label=f"desired_anchor_{batch}",
                    axis_colors=_DESIRED_FRAME_COLORS,
                )

                current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
                current_anchor_quat = self.robot_anchor_quat_w[batch]
                current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
                visualizer.add_frame(
                    position=current_anchor_pos,
                    rotation_matrix=current_rotation_matrix,
                    scale=0.15,
                    label=f"current_anchor_{batch}",
                )


@dataclass(kw_only=True)
class MultiMotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    entity_name: str
    motion_files: list[str] = field(default_factory=list)
    motion_path: str | None = None  # Alternative to motion_files: path to directory containing motion.npz files
    anchor_body_name: str
    body_names: tuple[str, ...]
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    

    fall_recovery_ratio: float = 0.0
    fall_recovery_pose_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "roll": (-math.pi / 2.0, math.pi / 2.0),
            "pitch": (-math.pi / 2.0, math.pi / 2.0),
            "yaw": (-math.pi, math.pi), 
        }
    )
    fall_recovery_velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    
    fall_recovery_joint_position_range: tuple[float, float] = (-0.52, 0.52)
    
    fall_recovery_joint_velocity_range: tuple[float, float] = (0.0, 0.0)

    # Ref Motion: Future/History steps configuration for N-step lookahead
    future_steps: int = 5 #1 
    history_steps: int = 5 #0

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    sampling_mode: Literal["adaptive", "uniform", "start"] = "uniform"

    # for downstream task training
    if_log_metrics: bool = True

    @dataclass
    class VizCfg:
        mode: Literal["ghost", "frames"] = "ghost"
        ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
        return MultiMotionCommand(self, env)
