# Claude Log — 已知隐患与 TODO

## [未修复] AMP lin_vel 坐标系偏差

**文件：** `src/mjlab/tasks/tracking/mdp/multi_commands.py` — `build_amp_obs_buffer()`

**问题：**
- Policy obs 的 `imu_lin_vel` 来自 MuJoCo `velocimeter` 传感器，挂载在 `imu_in_pelvis` site
  - site 位置：`pos="0.04525 0 -0.08339"`（距 pelvis CoM 约 9.5cm）
  - 返回的是 **site 处的线速度**，包含 `ω × r` 项：`v_site = v_com + ω × r_offset`
- Expert demo obs 的 `lin_vel_b` 来自 `_body_lin_vel_w_list`，是 **pelvis body CoM 的速度**，不含 `ω × r` 项

**量级：** 典型步行角速度 ~3 rad/s 时，差值约 0.2~0.3 m/s，不可忽视

**修法：** 在 `build_amp_obs_buffer()` 里对每帧 expert demo lin_vel 加上 `ω × r_offset`：
```python
r_offset = torch.tensor([0.04525, 0.0, -0.08339], device=self.device)
lin_vel_b += torch.cross(ang_vel_b, r_offset.expand_as(ang_vel_b), dim=-1)
```
注意 `r_offset` 需要已经在 pelvis 坐标系下（旋转后再加）。
