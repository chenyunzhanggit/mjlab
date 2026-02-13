"""Custom manager-based RL environment with extended functionality."""

from __future__ import annotations

from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, types
from mjlab.managers.command_manager import CommandManager
from mjlab.envs.manager_based_rl_env import cast

class CustomManagerBasedRlEnv(ManagerBasedRlEnv):
  """Custom environment that extends ManagerBasedRlEnv with additional functionality.
  
  This class allows you to override specific methods while maintaining compatibility
  with the base ManagerBasedRlEnv interface.
  """

  def __init__(
    self,
    cfg: ManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
    super().__init__(cfg, device, render_mode, **kwargs)


  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    action = action.to(self.device)
    if self.cfg.use_delta_action and isinstance(self.command_manager, CommandManager):
      try:
        motion_cfg = self.command_manager.get_term_cfg("motion")
        history_steps = getattr(motion_cfg, "history_steps", 0)
        joint_dim = self.action_manager.get_term("joint_pos").action_dim
        start = history_steps * joint_dim
        end = (history_steps + 1) * joint_dim
        policy_obs = cast(torch.Tensor, self.obs_buf["policy"])
        command_current_joint_pos = policy_obs[:, start:end]
        action = action * 0.25 + command_current_joint_pos
      except (AttributeError, KeyError, ValueError):
        pass
    
    self.action_manager.process_action(action)

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
      # Custom hook: apply curriculum forces during simulation
      self._apply_continuous_curriculum_forces()

    # Update env counters
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # Check terminations
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

    # Reset envs that terminated/timed-out
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()
      self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)

    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.obs_buf = self.observation_manager.compute(update_history=True)

    # Custom hook: update metrics after step
    self._update_custom_metrics()

    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    """Override reset method to add custom reset behavior.
    
    You can:
    - Initialize custom state for specific environments
    - Apply custom reset logic (e.g., set some envs to lying down)
    - Reset custom buffers or counters
    """
    # Your custom reset logic before base reset
    # For example, mark some environments as lying down
    # self._mark_lying_down_envs(env_ids)
    
    # Call parent reset method
    super()._reset_idx(env_ids)
    
    # Your custom reset logic after base reset
    # For example, initialize custom state
    # self._initialize_custom_state(env_ids)

  def _apply_continuous_curriculum_forces(self) -> None:
    """Apply continuous forces during simulation steps.
    
    This is called inside the decimation loop (every physics step).
    Use this for forces that need to be applied continuously, such as:
    - Pull forces for lying-down robots
    - Curriculum learning forces
    - Adaptive assistance forces
    
    Example:
        if hasattr(self, "_lying_down_mask") and hasattr(self, "scene"):
          lying_down_envs = torch.where(self._lying_down_mask)[0]
          if len(lying_down_envs) > 0:
            robot = self.scene["robot"]
            # Apply upward force (0, 0, 50N) to root body
            force = torch.zeros(len(lying_down_envs), 1, 3, device=self.device)
            force[:, 0, 2] = 50.0  # Upward force in z-direction
            torque = torch.zeros(len(lying_down_envs), 1, 3, device=self.device)
            robot.write_external_wrench_to_sim(force, torque, env_ids=lying_down_envs, body_ids=[0])
    """
    # Override this method to implement your custom curriculum logic
    pass

  def _update_custom_metrics(self) -> None:
    """Update custom metrics after each step.
    
    This is called after the base step() completes.
    Use this to track custom statistics or update logging.
    """
    # Your custom logic here
    # For example:
    # if hasattr(self, "_lying_down_mask"):
    #   num_lying = self._lying_down_mask.sum().item()
    #   self.extras["log"]["num_lying_down"] = num_lying
    pass
