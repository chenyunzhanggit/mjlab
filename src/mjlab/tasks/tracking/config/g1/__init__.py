from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.rl.amp_runner import AmpMotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.rl.distill_runner import (
  MotionTrackingDistillationRunner as MotionTrackingDistillationRunner,
)
from mjlab.tasks.tracking.rl.finetune_runner import (
  StudentTrackingFineTuneRunner as StudentTrackingFineTuneRunner,
)

from .env_cfgs import (
  unitree_g1_flat_tracking_env_cfg as unitree_g1_flat_tracking_env_cfg,
)
from .env_cfgs import (
  unitree_g1_student_env_cfg as unitree_g1_student_env_cfg,
)
from .env_cfgs import (
  unitree_g1_teacher_env_cfg as unitree_g1_teacher_env_cfg,
)
from .env_cfgs import (
  unitree_g1_teleoperation_amp_env_cfg as unitree_g1_teleoperation_amp_env_cfg,
)
from .env_cfgs import (
  unitree_g1_teleoperation_env_cfg as unitree_g1_teleoperation_env_cfg,
)
from .rl_cfg import (
  unitree_g1_student_distill_runner_cfg as unitree_g1_student_distill_runner_cfg,
)
from .rl_cfg import (
  unitree_g1_student_finetune_runner_cfg as unitree_g1_student_finetune_runner_cfg,
)
from .rl_cfg import (
  unitree_g1_teacher_ppo_runner_cfg as unitree_g1_teacher_ppo_runner_cfg,
)
from .rl_cfg import (
  unitree_g1_teleoperation_amp_runner_cfg as unitree_g1_teleoperation_amp_runner_cfg,
)
from .rl_cfg import (
  unitree_g1_teleoperation_ppo_runner_cfg as unitree_g1_teleoperation_ppo_runner_cfg,
)
from .rl_cfg import (
  unitree_g1_tracking_ppo_runner_cfg as unitree_g1_tracking_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)


register_mjlab_task(
  task_id="Mjlab-Tracking-Teleoperation-Unitree-G1",
  env_cfg=unitree_g1_teleoperation_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_teleoperation_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_teleoperation_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Teleoperation-AMP-Unitree-G1",
  env_cfg=unitree_g1_teleoperation_amp_env_cfg(
    has_state_estimation=False, disc_obs_steps=5
  ),
  play_env_cfg=unitree_g1_teleoperation_amp_env_cfg(
    has_state_estimation=False, play=True, disc_obs_steps=5
  ),
  rl_cfg=unitree_g1_teleoperation_amp_runner_cfg(),
  runner_cls=AmpMotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Teacher-Unitree-G1",
  env_cfg=unitree_g1_teacher_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_teacher_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_teacher_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Student-Unitree-G1",
  env_cfg=unitree_g1_student_env_cfg(),
  play_env_cfg=unitree_g1_student_env_cfg(play=True),
  rl_cfg=unitree_g1_student_distill_runner_cfg(),
  runner_cls=MotionTrackingDistillationRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Student-Finetune-Unitree-G1",
  env_cfg=unitree_g1_student_env_cfg(),
  play_env_cfg=unitree_g1_student_env_cfg(play=True),
  rl_cfg=unitree_g1_student_finetune_runner_cfg(),
  runner_cls=StudentTrackingFineTuneRunner,
)
