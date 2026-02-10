import torch
import rsl_rl.modules as _rsl_modules
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
    import ipdb; ipdb.set_trace()
    std = torch.clamp(std, min=0.001, max=0.5) 

    self.distribution = torch.distributions.Normal(mean, std)

_rsl_modules.MjlabActorCritic = MjlabActorCritic  # type: ignore[attr-defined]
