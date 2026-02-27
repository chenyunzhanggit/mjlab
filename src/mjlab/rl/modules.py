import rsl_rl.modules as _rsl_modules
import torch
from rsl_rl.modules.actor_critic import ActorCritic


class MjlabActorCritic(ActorCritic):
  def update_distribution(self, obs):
    # 直接复制原来的实现，然后只改你关心的几行就行
    if self.state_dependent_std:
      mean_and_std = self.actor(obs)
      if self.noise_std_type == "scalar":
        mean, std = torch.unbind(mean_and_std, dim=-2)
      elif self.noise_std_type == "log":
        mean, log_std = torch.unbind(mean_and_std, dim=-2)
        std = torch.exp(log_std)
      else:
        raise ValueError(
          f"Unknown standard deviation type: {self.noise_std_type}. "
          "Should be 'scalar' or 'log'"
        )
    else:
      mean = self.actor(obs)
      if self.noise_std_type == "scalar":
        std = self.std.expand_as(mean)
      elif self.noise_std_type == "log":
        std = torch.exp(self.log_std).expand_as(mean)
      else:
        raise ValueError(
          f"Unknown standard deviation type: {self.noise_std_type}. "
          "Should be 'scalar' or 'log'"
        )

    # 可变的 std 上界：
    # - 训练初期使用固定上界 0.5（与原来一致）
    # - 当外部把 self.max_pull_force 置为 0 后，每次调用 update_distribution
    #   就把 std_max 乘以 0.998，实现逐步减小噪声上界
    if not hasattr(self, "std_max"):
      # 初始 std 上界
      self.std_max = 1.0
    # 当环境的 max_pull_force 被设置为 0 时，开始衰减 std 上界
    if getattr(self, "max_pull_force", None) == 0:
      self.std_max *= 0.998
      # std_max 不能小于 0.5
      if self.std_max < 0.5:
        self.std_max = 0.5

    std = torch.clamp(std, min=0.001, max=self.std_max)  # from SONIC, with adjustable max

    self.distribution = torch.distributions.Normal(mean, std)


_rsl_modules.MjlabActorCritic = MjlabActorCritic  # type: ignore[attr-defined]
