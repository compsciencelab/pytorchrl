from gymnasium.core import Wrapper


class RewardShapeWrapper(Wrapper):
    """
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.
    """

    def __init__(self, env, positions=[], scale=1.0):
        self.target_positions = [(env.width - 2, 1)]  # positions
        self.scale = 0.5  # scale
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.env.agent_pos in self.target_positions:
            reward *= self.scale
        return obs, reward, terminated, truncated, info


class Gymnasium2GymWrapper(Wrapper):
    """
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, info

