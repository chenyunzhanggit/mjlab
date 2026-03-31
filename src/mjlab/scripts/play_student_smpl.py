"""Play a trained student policy driven by SMPL motion data.

Reads an SMPL .npz file (poses, trans, betas, mocap_framerate), runs
forward kinematics to obtain world-space joint positions & orientations,
converts to the robot-NPZ format expected by MultiMotionCommand, and
launches the student environment.

Usage
-----
python src/mjlab/scripts/play_student_smpl.py \
    --checkpoint-file student-ckp \
    --smpl-npz /path/to/pred_1.npz \
    --num-envs 1
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner
from scipy.spatial.transform import Rotation as R

import mjlab
import mjlab.tasks  # noqa: F401 — populates task registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.rl.exporter import export_policy_as_onnx
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

_STUDENT_TASK = "Mjlab-Tracking-Student-Unitree-G1"

# ── SMPL model helpers ────────────────────────────────────────────────────

_SMPL_MODEL_PATH = (
  "/home/zhengjk/PyProject/tele_MLD/deps/smpl_models/smpl/SMPL_MALE.pkl"
)


def _load_smpl_model(path: str) -> dict:
  """Load SMPL .pkl with a minimal chumpy stub (no chumpy install needed)."""
  pkg = types.ModuleType("chumpy")
  pkg.__path__ = []
  pkg.__package__ = "chumpy"

  class Ch:
    pass

  pkg.Ch = Ch
  for sub in [
    "ch",
    "utils",
    "reordering",
    "logic_ch",
    "linalg_squared",
    "ch_ops",
  ]:
    mod = types.ModuleType(f"chumpy.{sub}")
    mod.Ch = Ch
    sys.modules[f"chumpy.{sub}"] = mod
  sys.modules["chumpy"] = pkg
  with open(path, "rb") as f:
    return pickle.load(f, encoding="latin1")


def _aa_to_mat(aa: np.ndarray) -> np.ndarray:
  """(3,) axis-angle → (3,3) rotation matrix (Rodrigues)."""
  theta = float(np.linalg.norm(aa))
  if theta < 1e-8:
    return np.eye(3)
  k = aa / theta
  K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
  return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _rotmat_to_quat_wxyz(R_mat: np.ndarray) -> np.ndarray:
  """(3,3) rotation matrix → (4,) quaternion in (w,x,y,z) order."""
  q = R.from_matrix(R_mat).as_quat()  # scipy returns (x,y,z,w)
  return np.array([q[3], q[0], q[1], q[2]])


def smpl_fk(
  poses: np.ndarray,
  trans: np.ndarray,
  J_rest: np.ndarray,
  parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """SMPL forward kinematics.

  Parameters
  ----------
  poses : (T, 24, 3)  axis-angle per joint
  trans : (T, 3)       root translation (Y-up)
  J_rest : (24, 3)     rest-pose joint positions
  parents : (24,)      parent indices (-1 for root)

  Returns
  -------
  positions : (T, 24, 3)  world-space joint positions (Y-up)
  rotations : (T, 24, 3, 3)  world-space rotation matrices
  """
  T = poses.shape[0]
  positions = np.zeros((T, 24, 3), dtype=np.float64)
  rotations = np.zeros((T, 24, 3, 3), dtype=np.float64)

  for t in range(T):
    G = np.zeros((24, 4, 4))
    for j in range(24):
      Rl = _aa_to_mat(poses[t, j])
      if parents[j] < 0:
        Tmat = np.eye(4)
        Tmat[:3, :3] = Rl
        Tmat[:3, 3] = J_rest[j] + trans[t]
        G[j] = Tmat
      else:
        p = parents[j]
        Tl = np.eye(4)
        Tl[:3, :3] = Rl
        Tl[:3, 3] = J_rest[j] - J_rest[p]
        G[j] = G[p] @ Tl
    positions[t] = G[:, :3, 3]
    rotations[t] = G[:, :3, :3]

  return positions, rotations


# ── Coordinate conversion Y-up → Z-up ────────────────────────────────────

_YUP_TO_ZUP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)


def _convert_positions_yup_to_zup(pos: np.ndarray) -> np.ndarray:
  """Convert positions from Y-up to Z-up: (x, y, z) → (x, -z, y)."""
  return pos @ _YUP_TO_ZUP.T


def _convert_rotations_yup_to_zup(rot: np.ndarray) -> np.ndarray:
  """Convert rotation matrices from Y-up to Z-up frame."""
  # R_zup = C @ R_yup @ C^T  where C is the coordinate transform
  return _YUP_TO_ZUP @ rot @ _YUP_TO_ZUP.T


# ── Build robot-format NPZ from SMPL FK results ─────────────────────────

# IsaacLab body order (30 bodies)
ISAACLAB_BODY_NAMES = [
  "pelvis",
  "left_hip_pitch_link",
  "right_hip_pitch_link",
  "waist_yaw_link",
  "left_hip_roll_link",
  "right_hip_roll_link",
  "waist_roll_link",
  "left_hip_yaw_link",
  "right_hip_yaw_link",
  "torso_link",
  "left_knee_link",
  "right_knee_link",
  "left_shoulder_pitch_link",
  "right_shoulder_pitch_link",
  "left_ankle_pitch_link",
  "right_ankle_pitch_link",
  "left_shoulder_roll_link",
  "right_shoulder_roll_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
  "left_wrist_roll_link",
  "right_wrist_roll_link",
  "left_wrist_pitch_link",
  "right_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
]

# Height scale: G1 standing pelvis height / SMPL standing pelvis height
_G1_PELVIS_HEIGHT = 0.76

# SMPL joint → list of IsaacLab body indices to fill
# Anchor bodies (pelvis/torso) are scaled; hand/foot bodies keep absolute positions.
_SMPL_ANCHOR_INDICES: dict[int, list[int]] = {
  0: [
    ISAACLAB_BODY_NAMES.index("pelvis"),
    ISAACLAB_BODY_NAMES.index("torso_link"),
  ],
}
_SMPL_ENDPOINT_INDICES: dict[int, list[int]] = {
  7: [ISAACLAB_BODY_NAMES.index("left_ankle_roll_link")],
  8: [ISAACLAB_BODY_NAMES.index("right_ankle_roll_link")],
  20: [ISAACLAB_BODY_NAMES.index("left_wrist_yaw_link")],
  21: [ISAACLAB_BODY_NAMES.index("right_wrist_yaw_link")],
}


def _compute_lin_vel(pos: np.ndarray, fps: float) -> np.ndarray:
  """Compute linear velocity via central finite differences. (T, ..., 3) → (T, ..., 3)."""
  dt = 1.0 / fps
  vel = np.zeros_like(pos)
  vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
  vel[0] = (pos[1] - pos[0]) / dt
  vel[-1] = (pos[-1] - pos[-2]) / dt
  return vel


def _compute_ang_vel(rot: np.ndarray, fps: float) -> np.ndarray:
  """Compute angular velocity from rotation matrices. (T, 3, 3) → (T, 3)."""
  dt = 1.0 / fps
  T = rot.shape[0]
  ang_vel = np.zeros((T, 3), dtype=np.float64)
  for t in range(T):
    if t == 0:
      dR = rot[1] @ rot[0].T
    elif t == T - 1:
      dR = rot[t] @ rot[t - 1].T
    else:
      dR = rot[t + 1] @ rot[t - 1].T
      dt_eff = 2 * dt
      r = R.from_matrix(dR)
      ang_vel[t] = r.as_rotvec() / dt_eff
      continue
    r = R.from_matrix(dR)
    ang_vel[t] = r.as_rotvec() / dt
  return ang_vel


def build_robot_npz(
  smpl_npz_path: str,
  smpl_model_path: str,
  output_path: str,
) -> None:
  """Convert SMPL .npz → robot-format .npz for MultiMotionCommand."""
  # Load SMPL data
  data = np.load(smpl_npz_path)
  poses = data["poses"].astype(np.float64)  # (T, 24, 3)
  trans = data["trans"].astype(np.float64)  # (T, 3)
  fps = float(data["mocap_framerate"])

  # Load SMPL model
  smpl = _load_smpl_model(smpl_model_path)
  J_rest = np.array(smpl["J"], dtype=np.float64)  # (24, 3)
  kt = np.array(smpl["kintree_table"])
  parents = kt[0].astype(int)
  parents[0] = -1

  # FK → world-space positions & rotations (Y-up)
  print(f"[INFO] Running SMPL FK on {poses.shape[0]} frames ...")
  positions_yup, rotations_yup = smpl_fk(poses, trans, J_rest, parents)

  # Convert Y-up → Z-up
  T = positions_yup.shape[0]
  positions = np.zeros_like(positions_yup)
  rotations = np.zeros_like(rotations_yup)
  for j in range(24):
    positions[:, j] = _convert_positions_yup_to_zup(positions_yup[:, j])
    for t in range(T):
      rotations[t, j] = _convert_rotations_yup_to_zup(rotations_yup[t, j])

  # Ground align: shift each frame so the lowest joint sits at z=0
  ground_z = positions[:, :, 2].min(axis=1, keepdims=True)  # (T, 1)
  positions[:, :, 2] -= ground_z  # broadcast (T,24) -= (T,1)

  # Build robot body arrays (30 bodies)
  num_bodies = 30
  body_pos_w = np.zeros((T, num_bodies, 3), dtype=np.float32)
  body_quat_w = np.tile(
    np.array([1, 0, 0, 0], dtype=np.float32), (T, num_bodies, 1)
  )  # identity quat (w,x,y,z)
  body_lin_vel_w = np.zeros((T, num_bodies, 3), dtype=np.float32)
  body_ang_vel_w = np.zeros((T, num_bodies, 3), dtype=np.float32)

  # ── Anchor (torso_link / pelvis): real SMPL data with height scale ───
  smpl_root = 0  # SMPL pelvis
  # Compute scale from first frame's actual pelvis height (Z-up, so z-coord)
  smpl_pelvis_height = float(positions[0, smpl_root, 2])
  height_scale = (
    _G1_PELVIS_HEIGHT / smpl_pelvis_height if smpl_pelvis_height > 0.1 else 1.0
  )
  print(f"[INFO] SMPL pelvis height={smpl_pelvis_height:.4f}, scale={height_scale:.4f}")
  anchor_pos = positions[:, smpl_root].copy()  # (T, 3)
  anchor_pos[:, 2] *= height_scale  # only scale height (z in Z-up)
  anchor_lin_vel = _compute_lin_vel(anchor_pos, fps)

  # Anchor rotation: extract yaw-only quaternion from SMPL root.
  # This is needed so that subtract_frame_transforms in motion_ref_hand_pos_b
  # computes hand positions in the anchor's body frame (consistent with training).
  # Pitch/roll are zeroed out — the policy controls those.
  anchor_rot = rotations[:, smpl_root]  # (T, 3, 3)
  anchor_ang_vel = _compute_ang_vel(anchor_rot, fps)  # (T, 3)
  anchor_quat = np.zeros((T, 4), dtype=np.float32)
  for t in range(T):
    # Extract yaw angle from rotation matrix (Z-up: yaw = rotation around z-axis)
    r = R.from_matrix(anchor_rot[t])
    yaw = r.as_euler("ZYX")[0]  # intrinsic ZYX → first component is yaw
    anchor_quat[t] = R.from_euler("Z", yaw).as_quat()  # scipy (x,y,z,w)
    # Convert scipy (x,y,z,w) → (w,x,y,z)
    anchor_quat[t] = np.array(
      [anchor_quat[t, 3], anchor_quat[t, 0], anchor_quat[t, 1], anchor_quat[t, 2]]
    )

  for il_idx in _SMPL_ANCHOR_INDICES[smpl_root]:
    body_pos_w[:, il_idx] = anchor_pos.astype(np.float32)
    body_quat_w[:, il_idx] = anchor_quat
    body_lin_vel_w[:, il_idx] = anchor_lin_vel.astype(np.float32)
    body_ang_vel_w[:, il_idx] = anchor_ang_vel.astype(np.float32)

  # ── Hands: absolute SMPL wrist world positions (no scaling) ─────────
  # motion_ref_hand_pos_b computes R_anchor^T @ (hand - anchor).
  # When robot tracks anchor (pos + yaw), robot hand reaches hand's absolute position.
  _SMPL_HAND_MAP = {
    20: ISAACLAB_BODY_NAMES.index("left_wrist_yaw_link"),
    21: ISAACLAB_BODY_NAMES.index("right_wrist_yaw_link"),
  }
  for smpl_idx, il_idx in _SMPL_HAND_MAP.items():
    hand_pos = positions[:, smpl_idx]  # (T, 3), absolute world position
    body_pos_w[:, il_idx] = hand_pos.astype(np.float32)
    body_quat_w[:, il_idx] = anchor_quat
    body_lin_vel_w[:, il_idx] = _compute_lin_vel(hand_pos, fps).astype(np.float32)
    body_name = ISAACLAB_BODY_NAMES[il_idx]
    print(
      f"[DEBUG] SMPL {body_name} frame0: pos={hand_pos[0]}, z(height)={hand_pos[0, 2]:.4f}"
    )
  print(
    f"[DEBUG] SMPL anchor frame0: pos={anchor_pos[0]}, z(height)={anchor_pos[0, 2]:.4f}"
  )

  # ── Feet: fixed offsets relative to anchor in anchor body frame ──────
  _DEBUG_FOOT_OFFSETS: dict[str, np.ndarray] = {
    "left_ankle_roll_link": np.array([0.0, 0.1, -0.76], dtype=np.float64),
    "right_ankle_roll_link": np.array([0.0, -0.1, -0.76], dtype=np.float64),
  }
  for body_name, offset in _DEBUG_FOOT_OFFSETS.items():
    il_idx = ISAACLAB_BODY_NAMES.index(body_name)
    foot_world = np.zeros((T, 3), dtype=np.float64)
    for t in range(T):
      yaw_rot = R.from_quat(
        [anchor_quat[t, 1], anchor_quat[t, 2], anchor_quat[t, 3], anchor_quat[t, 0]]
      ).as_matrix()
      foot_world[t] = anchor_pos[t] + yaw_rot @ offset
    body_pos_w[:, il_idx] = foot_world.astype(np.float32)
    body_quat_w[:, il_idx] = anchor_quat
    body_lin_vel_w[:, il_idx] = _compute_lin_vel(foot_world, fps).astype(np.float32)

  # Dummy joint data (not used by student observations)
  joint_pos = np.zeros((T, 29), dtype=np.float32)
  joint_vel = np.zeros((T, 29), dtype=np.float32)

  np.savez(
    output_path,
    fps=np.float32(fps),
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
  )
  # ── Debug: print anchor data for first few frames ───────────────────
  torso_idx = ISAACLAB_BODY_NAMES.index("torso_link")
  print("[DEBUG] Anchor (torso_link) data for first 5 frames:")
  for t in range(min(5, T)):
    p = body_pos_w[t, torso_idx]
    v = body_lin_vel_w[t, torso_idx]
    av = body_ang_vel_w[t, torso_idx]
    q = body_quat_w[t, torso_idx]
    print(f"  t={t}: pos={p}, quat={q}, lin_vel={v}, ang_vel={av}")
  print(f"[DEBUG] fps={fps}")
  print(f"[INFO] Saved robot-format NPZ: {output_path}")
  print(f"       frames={T}, fps={fps}, body_pos_w={body_pos_w.shape}")


# ── Main ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlayStudentSmplConfig:
  checkpoint_file: str | None = None
  """Path to the student model checkpoint (.pt file)."""

  wandb_run_path: str | None = None
  """WandB run path (entity/project/run_id) to download the checkpoint from."""

  smpl_npz: str = "/home/zhengjk/PyProject/QwenFinetune/data/pred_1.npz"
  """Path to the SMPL .npz file (poses, trans, betas, mocap_framerate)."""

  smpl_model: str = _SMPL_MODEL_PATH
  """Path to SMPL_MALE.pkl model file."""

  num_envs: int = 1
  device: str | None = None

  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None

  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False


def main() -> None:
  cfg = tyro.cli(PlayStudentSmplConfig, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # ── Convert SMPL → robot NPZ ──────────────────────────────────────────
  tmp_dir = tempfile.mkdtemp(prefix="smpl2robot_")
  robot_npz_path = os.path.join(tmp_dir, "smpl_converted.npz")
  build_robot_npz(cfg.smpl_npz, cfg.smpl_model, robot_npz_path)

  # ── Environment config ────────────────────────────────────────────────
  env_cfg = load_env_cfg(_STUDENT_TASK, play=True)
  agent_cfg = load_rl_cfg(_STUDENT_TASK)

  from mjlab.tasks.tracking.mdp.multi_commands import MultiMotionCommandCfg

  if "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MultiMotionCommandCfg
  ):
    motion_cmd = env_cfg.commands["motion"]
    motion_cmd.motion_files = [robot_npz_path]
    print(f"[INFO] Using converted SMPL motion: {robot_npz_path}")

  # ── Terminations ──────────────────────────────────────────────────────
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO] Terminations disabled")

  # ── Checkpoint ────────────────────────────────────────────────────────
  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
  elif cfg.wandb_run_path is not None:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    resume_path, was_cached = get_wandb_checkpoint_path(
      log_root_path, Path(cfg.wandb_run_path)
    )
    print(
      f"[INFO] Checkpoint: {resume_path.name} "
      f"({'cached' if was_cached else 'downloaded'})"
    )
  else:
    raise ValueError("Provide --checkpoint-file or --wandb-run-path.")

  log_dir = resume_path.parent

  # ── Environment ───────────────────────────────────────────────────────
  env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if cfg.video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if cfg.video:
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # ── Load student runner & policy ──────────────────────────────────────
  runner_cls = load_runner_cls(_STUDENT_TASK) or OnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  # ── Export ONNX ───────────────────────────────────────────────────────
  export_model_dir = str(log_dir / "exported")
  ckpt_num = resume_path.stem.split("_")[-1]
  export_policy_as_onnx(
    env=env.unwrapped,
    actor_critic=runner.alg.policy,
    path=export_model_dir,
    normalizer=runner.alg.policy.actor_obs_normalizer
    if runner.alg.policy.actor_obs_normalization
    else None,
    filename=f"student_{ckpt_num}.onnx",
    verbose=False,
  )

  # ── Viewer ────────────────────────────────────────────────────────────
  obs = env.get_observations()
  # Debug: print student observation breakdown
  raw_env = env.unwrapped
  from mjlab.tasks.tracking.mdp.multi_commands import MultiMotionCommand

  cmd: MultiMotionCommand = raw_env.command_manager.get_term("motion")  # type: ignore[assignment]
  print("[DEBUG] === Student observation at step 0 ===")
  print(f"  anchor_pos_w (from cmd): {cmd.anchor_pos_w[0].cpu().numpy()}")
  print(f"  anchor_height: {cmd.anchor_pos_w[0, 2].item():.4f}")
  print(f"  anchor_lin_vel_w shape: {cmd.anchor_lin_vel_w.shape}")
  print(f"  anchor_lin_vel_w[0]: {cmd.anchor_lin_vel_w[0].cpu().numpy()}")
  print(f"  anchor_ang_vel_w[0]: {cmd.anchor_ang_vel_w[0].cpu().numpy()}")
  print(f"  body_pos_w[0]: {cmd.body_pos_w[0].cpu().numpy()}")
  print(f"  body_quat_w[0]: {cmd.body_quat_w[0].cpu().numpy()}")
  for k, v in obs.items():
    if hasattr(v, "shape"):
      print(f"  obs[{k}] shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}")
    else:
      for k2, v2 in v.items():
        print(
          f"  obs[{k}][{k2}] shape={v2.shape}, min={v2.min():.4f}, max={v2.max():.4f}"
        )
  # ── Debug visualization: draw SMPL wrist reference points ───────────
  _orig_update_vis = raw_env.update_visualizers

  # hand body indices within the tracked body_names list
  hand_body_indices = [
    cmd.cfg.body_names.index("left_wrist_yaw_link"),
    cmd.cfg.body_names.index("right_wrist_yaw_link"),
  ]

  _vis_step_counter = [0]

  def _update_vis_with_hands(visualizer):  # type: ignore[no-untyped-def]
    _orig_update_vis(visualizer)
    env_origin = raw_env.scene.env_origins[0].cpu().numpy()

    # Draw reference hand positions as colored spheres
    ref_body_pos = cmd.body_pos_w[0]  # (num_bodies, 3)
    anchor_idx = cmd.motion_anchor_body_index
    ref_anchor_pos = ref_body_pos[anchor_idx].cpu().numpy()

    colors = [(1.0, 0.0, 0.0, 0.8), (0.0, 0.0, 1.0, 0.8)]  # red=left, blue=right
    labels = ["L_wrist_ref", "R_wrist_ref"]
    for i, bidx in enumerate(hand_body_indices):
      pos = ref_body_pos[bidx].cpu().numpy()
      visualizer.add_sphere(
        center=pos + env_origin,
        radius=0.03,
        color=colors[i],
        label=labels[i],
      )

    # Print debug info every 30 steps
    if _vis_step_counter[0] % 30 == 0:
      from mjlab.tasks.tracking.mdp.student_observations import (
        motion_ref_ang_vel_current,
        motion_ref_hand_pos_b,
        motion_ref_vel_current,
      )

      hand_pos_b = motion_ref_hand_pos_b(raw_env, "motion")
      ref_lin_vel = motion_ref_vel_current(raw_env, "motion")
      ref_ang_vel = motion_ref_ang_vel_current(raw_env, "motion")
      # Get robot wrist positions from body link pose (pos+quat)
      robot_asset = raw_env.scene["robot"]
      left_wrist_idx = robot_asset.find_bodies("left_wrist_yaw_link")[0][0]
      right_wrist_idx = robot_asset.find_bodies("right_wrist_yaw_link")[0][0]
      robot_lw_pos = (
        robot_asset.data.body_link_pose_w[0, left_wrist_idx, :3].cpu().numpy()
      )
      robot_rw_pos = (
        robot_asset.data.body_link_pose_w[0, right_wrist_idx, :3].cpu().numpy()
      )
      # SMPL ref wrist positions from command buffer
      smpl_lw_idx = cmd.cfg.body_names.index("left_wrist_yaw_link")
      smpl_rw_idx = cmd.cfg.body_names.index("right_wrist_yaw_link")
      smpl_lw_pos = cmd.body_pos_w[0, smpl_lw_idx].cpu().numpy()
      smpl_rw_pos = cmd.body_pos_w[0, smpl_rw_idx].cpu().numpy()
      print(f"[DEBUG step={_vis_step_counter[0]}]")
      print(f"  SMPL left_wrist  pos={smpl_lw_pos}, z={smpl_lw_pos[2]:.4f}")
      print(f"  Robot left_wrist pos={robot_lw_pos}, z={robot_lw_pos[2]:.4f}")
      print(f"  SMPL right_wrist pos={smpl_rw_pos}, z={smpl_rw_pos[2]:.4f}")
      print(f"  Robot right_wrist pos={robot_rw_pos}, z={robot_rw_pos[2]:.4f}")
      print(f"  ref anchor pos: {ref_anchor_pos}")
      print(f"  ref lin_vel: {ref_lin_vel[0].cpu().numpy()}")
      print(f"  ref ang_vel: {ref_ang_vel[0].cpu().numpy()}")
      print(f"  hand_pos_b: {hand_pos_b[0].cpu().numpy()}")
    _vis_step_counter[0] += 1

  raw_env.update_visualizers = _update_vis_with_hands

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer: {resolved_viewer}")

  env.close()


if __name__ == "__main__":
  main()
