import math
from dataclasses import dataclass, field
from typing import Any, cast

import mujoco
import numpy as np
import torch
import warp as wp
from prettytable import PrettyTable

from mjlab.envs import types
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.action_manager import ActionManager, ActionTermCfg
from mjlab.managers.command_manager import (
  CommandManager,
  CommandTermCfg,
  NullCommandManager,
)
from mjlab.managers.curriculum_manager import (
  CurriculumManager,
  CurriculumTermCfg,
  NullCurriculumManager,
)
from mjlab.managers.event_manager import EventManager, EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationManager
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlab.scene import Scene
from mjlab.scene.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.utils import random as random_utils
from mjlab.utils.lab_api.math import sample_uniform
from mjlab.utils.logging import print_info
from mjlab.utils.spaces import Box
from mjlab.utils.spaces import Dict as DictSpace
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg:
  """Configuration for a manager-based RL environment.

  This config defines all aspects of an RL environment: the physical scene,
  observations, actions, rewards, terminations, and optional features like
  commands and curriculum learning.

  The environment step size is ``sim.mujoco.timestep * decimation``. For example,
  with a 2ms physics timestep and decimation=10, the environment runs at 50Hz.
  """

  # Base environment configuration.

  decimation: int
  """Number of physics simulation steps per environment step. Higher values mean
  coarser control frequency. Environment step duration = physics_dt * decimation."""

  scene: SceneCfg
  """Scene configuration defining terrain, entities, and sensors. The scene
  specifies ``num_envs``, the number of parallel environments."""

  observations: dict[str, ObservationGroupCfg] = field(default_factory=dict)
  """Observation groups configuration. Each group (e.g., "policy", "critic") contains
  observation terms that are concatenated. Groups can have different settings for
  noise, history, and delay."""

  actions: dict[str, ActionTermCfg] = field(default_factory=dict)
  """Action terms configuration. Each term controls a specific entity/aspect
  (e.g., joint positions). Action dimensions are concatenated across terms."""

  events: dict[str, EventTermCfg] = field(
    default_factory=lambda: {
      "reset_scene_to_default": EventTermCfg(
        func=reset_scene_to_default,
        mode="reset",
      )
    }
  )
  """Event terms for domain randomization and state resets. Default includes
  ``reset_scene_to_default`` which resets entities to their initial state.
  Can be set to empty to disable all events including default reset."""

  seed: int | None = None
  """Random seed for reproducibility. If None, a random seed is used. The actual
  seed used is stored back into this field after initialization."""

  sim: SimulationCfg = field(default_factory=SimulationCfg)
  """Simulation configuration including physics timestep, solver iterations,
  contact parameters, and NaN guarding."""

  viewer: ViewerConfig = field(default_factory=ViewerConfig)
  """Viewer configuration for rendering (camera position, resolution, etc.)."""

  # RL-specific configuration.

  episode_length_s: float = 0.0
  """Duration of an episode (in seconds).

  Episode length in steps is computed as:
    ceil(episode_length_s / (sim.mujoco.timestep * decimation))
  """

  rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
  """Reward terms configuration."""

  terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
  """Termination terms configuration. If empty, episodes never reset. Use
  ``mdp.time_out`` with ``time_out=True`` for episode time limits."""

  commands: dict[str, CommandTermCfg] = field(default_factory=dict)
  """Command generator terms (e.g., velocity targets)."""

  curriculum: dict[str, CurriculumTermCfg] = field(default_factory=dict)
  """Curriculum terms for adaptive difficulty."""

  is_finite_horizon: bool = False
  """Whether the task has a finite or infinite horizon. Defaults to False (infinite).

  - **Finite horizon (True)**: The time limit defines the task boundary. When reached,
    no future value exists beyond it, so the agent receives a terminal done signal.
  - **Infinite horizon (False)**: The time limit is an artificial cutoff. The agent
    receives a truncated done signal to bootstrap the value of continuing beyond the
    limit.
  """

  scale_rewards_by_dt: bool = True
  """Whether to multiply rewards by the environment step duration (dt).

  When True (default), reward values are scaled by step_dt to normalize cumulative
  episodic rewards across different simulation frequencies. Set to False for
  algorithms that expect unscaled reward signals (e.g., HER, static reward scaling).
  """

  use_delta_action: bool = False
  """Whether to use delta action mode for joint position control.

  When True, network actions are treated as delta actions and added to command_joint_pos
  before being processed. This is useful for tracking tasks where the policy outputs
  relative changes to the reference motion. Defaults to False (absolute actions).
  """


class ManagerBasedRlEnv:
  """Manager-based RL environment."""

  is_vector_env = True
  metadata = {
    "render_modes": [None, "rgb_array"],
    "mujoco_version": mujoco.__version__,
    "warp_version": wp.config.version,
  }
  cfg: ManagerBasedRlEnvCfg

  def __init__(
    self,
    cfg: ManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
    # Initialize base environment state.
    self.cfg = cfg
    if self.cfg.seed is not None:
      self.cfg.seed = self.seed(self.cfg.seed)
    self._sim_step_counter = 0
    self.extras = {}
    self.obs_buf = {}

    # Initialize scene and simulation.
    self.scene = Scene(self.cfg.scene, device=device)
    self.sim = Simulation(
      num_envs=self.scene.num_envs,
      cfg=self.cfg.sim,
      model=self.scene.compile(),
      device=device,
    )

    self.scene.initialize(
      mj_model=self.sim.mj_model,
      model=self.sim.model,
      data=self.sim.data,
    )

    # Print environment info.
    print_info("")
    table = PrettyTable()
    table.title = "Base Environment"
    table.field_names = ["Property", "Value"]
    table.align["Property"] = "l"
    table.align["Value"] = "l"
    table.add_row(["Number of environments", self.num_envs])
    table.add_row(["Environment device", self.device])
    table.add_row(["Environment seed", self.cfg.seed])
    table.add_row(["Physics step-size", self.physics_dt])
    table.add_row(["Environment step-size", self.step_dt])
    print_info(table.get_string())
    print_info("")

    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(
      cfg.scene.num_envs, device=device, dtype=torch.long
    )
    # Initialize max pull force for fall recovery curriculum
    self.max_pull_force = 200.0  # Initial max pull force (Newtons)
    self.render_mode = render_mode
    self._offline_renderer: OffscreenRenderer | None = None
    if self.render_mode == "rgb_array":
      renderer = OffscreenRenderer(
        model=self.sim.mj_model, cfg=self.cfg.viewer, scene=self.scene
      )
      renderer.initialize()
      self._offline_renderer = renderer
    self.metadata["render_fps"] = 1.0 / self.step_dt

    # Load all managers.
    self.load_managers()
    self.setup_manager_visualizers()

  # Properties.

  @property
  def num_envs(self) -> int:
    """Number of parallel environments."""
    return self.scene.num_envs

  @property
  def physics_dt(self) -> float:
    """Physics simulation step size."""
    return self.cfg.sim.mujoco.timestep

  @property
  def step_dt(self) -> float:
    """Environment step size (physics_dt * decimation)."""
    return self.cfg.sim.mujoco.timestep * self.cfg.decimation

  @property
  def device(self) -> str:
    """Device for computation."""
    return self.sim.device

  @property
  def max_episode_length_s(self) -> float:
    """Maximum episode length in seconds."""
    return self.cfg.episode_length_s

  @property
  def max_episode_length(self) -> int:
    """Maximum episode length in steps."""
    return math.ceil(self.max_episode_length_s / self.step_dt)

  @property
  def unwrapped(self) -> "ManagerBasedRlEnv":
    """Get the unwrapped environment (base case for wrapper chains)."""
    return self

  # Methods.

  def setup_manager_visualizers(self) -> None:
    self.manager_visualizers = {}
    if getattr(self.command_manager, "active_terms", None):
      self.manager_visualizers["command_manager"] = self.command_manager

  def load_managers(self) -> None:
    """Load and initialize all managers.

    Order is important! Event and command managers must be loaded first,
    then action and observation managers, then other RL managers.
    """
    # Event manager (required before everything else for domain randomization).
    self.event_manager = EventManager(self.cfg.events, self)
    print_info(f"[INFO] {self.event_manager}")

    self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)

    # Command manager (must be before observation manager since observations
    # may reference commands).
    if len(self.cfg.commands) > 0:
      self.command_manager = CommandManager(self.cfg.commands, self)
    else:
      self.command_manager = NullCommandManager()
    print_info(f"[INFO] {self.command_manager}")

    # Action and observation managers.
    self.action_manager = ActionManager(self.cfg.actions, self)
    print_info(f"[INFO] {self.action_manager}")
    self.observation_manager = ObservationManager(self.cfg.observations, self)
    print_info(f"[INFO] {self.observation_manager}")

    # Other RL-specific managers.

    self.termination_manager = TerminationManager(self.cfg.terminations, self)
    print_info(f"[INFO] {self.termination_manager}")
    self.reward_manager = RewardManager(
      self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.reward_manager}")
    if len(self.cfg.curriculum) > 0:
      self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
    else:
      self.curriculum_manager = NullCurriculumManager()
    print_info(f"[INFO] {self.curriculum_manager}")

    # Configure spaces for the environment.
    self._configure_gym_env_spaces()

    # Initialize startup events if defined.
    if "startup" in self.event_manager.available_modes:
      self.event_manager.apply(mode="startup")

  def _update_pull_force_curriculum(
    self,
    avg_height_diff: float,
  ) -> None:
    """Update the max pull force curriculum based on average height difference.

    Args:
      avg_height_diff: Average height difference between target and actual height
        for fall recovery environments.
    """
    # Adaptive curriculum: adjust max_pull_force based on average height difference
    if avg_height_diff > 0.4:
      # If average height difference > 0.4, keep max force at 200
      self.max_pull_force = 200.0
    elif avg_height_diff < 0.4:
      # If average height difference < 0.4, reduce max force by 0.99
      self.max_pull_force = self.max_pull_force * 0.998
      if avg_height_diff >= 0.3:
        # When 0.3 <= avg_height_diff < 0.4, enforce lower limit of 120
        if self.max_pull_force < 120.0:
          self.max_pull_force = 120.0
      else:
        # If average height difference < 0.3, continue reducing by 0.99
        # No lower limit after this point
        self.max_pull_force = self.max_pull_force * 0.998

    # Log max_pull_force to extras
    if "log" not in self.extras:
      self.extras["log"] = {}
    self.extras["log"]["Curriculum/max_pull_force"] = self.max_pull_force

  def _apply_fall_recovery_pull_force(
    self,
    motion_cfg: Any,
  ) -> float | None:
    """Apply pull force to fall recovery environments.

    Args:
      motion_cfg: Motion command configuration.

    Returns:
      Average height difference if fall recovery environments exist, None otherwise.
    """
    try:
      # Get the command instance (not config) to access runtime data
      motion_cmd = self.command_manager.get_term("motion")
      if motion_cmd is None:
        return None

      # Check if this is a MultiMotionCommand with fall recovery support
      from mjlab.tasks.tracking.mdp.multi_commands import MultiMotionCommand

      if not isinstance(motion_cmd, MultiMotionCommand):
        return None

      # Get fall recovery mask from command instance
      fall_recovery_mask = getattr(motion_cmd, "init_fall_recovery_mask", None)
      if fall_recovery_mask is None:
        return None

      fall_recovery_env_ids = torch.where(fall_recovery_mask)[0]
      if len(fall_recovery_env_ids) == 0:
        return None

      # Get robot entity from command instance
      robot = motion_cmd.robot
      # Get base body index (use anchor body index) from command instance
      base_body_index = motion_cmd.robot_anchor_body_index
      # Get actual and target heights for fall recovery environments
      robot_anchor_pos_w = motion_cmd.robot_anchor_pos_w
      anchor_pos_w = motion_cmd.anchor_pos_w

      actual_height = robot_anchor_pos_w[fall_recovery_env_ids, 2]
      target_height = anchor_pos_w[fall_recovery_env_ids, 2]

      height_diff = target_height - actual_height
      height_threshold = 0.2
      need_force_mask = height_diff > height_threshold

      # [NOTE] 先全部拉起
      need_force_mask = torch.ones(
        len(fall_recovery_env_ids), dtype=torch.bool, device=self.device
      )

      # Calculate average height difference for fall recovery environments
      avg_height_diff = height_diff.mean().item()

      # Create force tensor for all fall recovery environments
      num_fall_recovery = len(fall_recovery_env_ids)
      force_tensor = torch.zeros((num_fall_recovery, 1, 3), device=self.device)

      # Sample random force values from 0 to max_pull_force
      if need_force_mask.any():
        num_need_force = int(need_force_mask.sum().item())
        # Sample random force values (0 to max_pull_force)
        random_force_values = sample_uniform(
          self.max_pull_force / 2.0,
          self.max_pull_force,
          (num_need_force,),
          device=self.device,
        )

        force_tensor[need_force_mask, 0, 2] = random_force_values

      # Apply zero torque
      torque_tensor = torch.zeros((num_fall_recovery, 1, 3), device=self.device)

      # for play
      # force_tensor = force_tensor * 0.0
      # torque_tensor = torque_tensor * 0.0
      # print("force_tensor: ", force_tensor)
      # print("torque_tensor: ", torque_tensor)

      robot.write_external_wrench_to_sim(
        force_tensor,
        torque_tensor,
        env_ids=fall_recovery_env_ids,
        body_ids=[base_body_index],
      )
      # print("force_tensor: ", force_tensor)
      return avg_height_diff
    except (AttributeError, KeyError, TypeError):
      # Command term might not exist or might not be MultiMotionCommand
      return None

  def reset(
    self,
    *,
    seed: int | None = None,
    env_ids: torch.Tensor | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[types.VecEnvObs, dict]:
    del options  # Unused.
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
    if seed is not None:
      self.seed(seed)
    self._reset_idx(env_ids)
    self.scene.write_data_to_sim()
    self.sim.forward()
    self.obs_buf = self.observation_manager.compute(update_history=True)
    return self.obs_buf, self.extras

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    action = action.to(self.device)
    # get the motion command configuration
    motion_cfg = self.command_manager.get_term_cfg("motion")
    if self.cfg.use_delta_action and isinstance(self.command_manager, CommandManager):
      try:
        # command_current_joint_pos = self.command_manager.get_command_joint_pos_current("motion")  # 到 compute_obs 的时候 current joint_pos command 是下一帧了
        history_steps = getattr(motion_cfg, "history_steps", 0)
        joint_dim = self.action_manager.get_term("joint_pos").action_dim
        start = history_steps * joint_dim
        end = (history_steps + 1) * joint_dim
        policy_obs = cast(torch.Tensor, self.obs_buf["policy"])
        command_current_joint_pos = policy_obs[:, start:end]
        # [NOTE] * 0.25 to accelerate the exploring process, as the output is the delta action on command.
        action = action * 0.25 + command_current_joint_pos
      except (AttributeError, KeyError, ValueError):
        pass

    self.action_manager.process_action(action)

    # Track average height difference for curriculum update
    avg_height_diff: float | None = None
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
      # Apply curriculum pull force for fall recovery environments
      if getattr(motion_cfg, "fall_recovery_ratio", 0) > 0.0:
        result = self._apply_fall_recovery_pull_force(motion_cfg)
        if result is not None:
          avg_height_diff = result
    # Update env counters.
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # Check terminations.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

    # Reset envs that terminated/timed-out and log the episode info.
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()
      self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)

    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.obs_buf = self.observation_manager.compute(update_history=True)
    # Update pull force curriculum if fall recovery environments exist
    if avg_height_diff is not None:
      # print("fall recovery ")
      self._update_pull_force_curriculum(avg_height_diff)

    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def render(self) -> np.ndarray | None:
    if self.render_mode == "human" or self.render_mode is None:
      return None
    elif self.render_mode == "rgb_array":
      if self._offline_renderer is None:
        raise ValueError("Offline renderer not initialized")
      debug_callback = (
        self.update_visualizers if hasattr(self, "update_visualizers") else None
      )
      self._offline_renderer.update(self.sim.data, debug_vis_callback=debug_callback)
      return self._offline_renderer.render()
    else:
      raise NotImplementedError(
        f"Render mode {self.render_mode} is not supported. "
        f"Please use: {self.metadata['render_modes']}."
      )

  def close(self) -> None:
    if self._offline_renderer is not None:
      self._offline_renderer.close()

  @staticmethod
  def seed(seed: int = -1) -> int:
    if seed == -1:
      seed = np.random.randint(0, 10_000)
    print_info(f"Setting seed: {seed}")
    random_utils.seed_rng(seed)
    return seed

  def update_visualizers(self, visualizer: DebugVisualizer) -> None:
    for mod in self.manager_visualizers.values():
      mod.debug_vis(visualizer)
    for sensor in self.scene.sensors.values():
      sensor.debug_vis(visualizer)

  # Private methods.

  def _configure_gym_env_spaces(self) -> None:
    from mjlab.utils.spaces import batch_space

    self.single_observation_space = DictSpace()
    for group_name, group_term_names in self.observation_manager.active_terms.items():
      has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
      group_dim = self.observation_manager.group_obs_dim[group_name]
      if has_concatenated_obs:
        assert isinstance(group_dim, tuple)
        self.single_observation_space.spaces[group_name] = Box(
          shape=group_dim, low=-math.inf, high=math.inf
        )
      else:
        assert not isinstance(group_dim, tuple)
        group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
        # Create a nested dict for this group.
        group_space = DictSpace()
        for term_name, term_dim, _term_cfg in zip(
          group_term_names, group_dim, group_term_cfgs, strict=False
        ):
          group_space.spaces[term_name] = Box(
            shape=term_dim, low=-math.inf, high=math.inf
          )
        self.single_observation_space.spaces[group_name] = group_space

    action_dim = sum(self.action_manager.action_term_dim)
    self.single_action_space = Box(shape=(action_dim,), low=-math.inf, high=math.inf)

    self.observation_space = batch_space(self.single_observation_space, self.num_envs)
    self.action_space = batch_space(self.single_action_space, self.num_envs)

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    self.curriculum_manager.compute(env_ids=env_ids)
    self.sim.reset(env_ids)
    self.scene.reset(env_ids)

    if "reset" in self.event_manager.available_modes:
      env_step_count = self._sim_step_counter // self.cfg.decimation
      self.event_manager.apply(
        mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
      )

    # NOTE: This is order sensitive.
    self.extras["log"] = dict()
    # observation manager.
    info = self.observation_manager.reset(env_ids)
    self.extras["log"].update(info)
    # action manager.
    info = self.action_manager.reset(env_ids)
    self.extras["log"].update(info)
    # rewards manager.
    info = self.reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    # curriculum manager.
    info = self.curriculum_manager.reset(env_ids)
    self.extras["log"].update(info)
    # command manager.
    info = self.command_manager.reset(env_ids)
    self.extras["log"].update(info)
    # event manager.
    info = self.event_manager.reset(env_ids)
    self.extras["log"].update(info)
    # termination manager.
    info = self.termination_manager.reset(env_ids)
    self.extras["log"].update(info)
    # reset the episode length buffer.
    self.episode_length_buf[env_ids] = 0
