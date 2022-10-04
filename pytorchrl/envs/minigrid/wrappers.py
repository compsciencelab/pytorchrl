import gym
import gymnasium
from gymnasium.core import Wrapper


class RewardShapeWrapper(Wrapper):
    """Wrapper to change certain reward values at specific positions."""

    def __init__(self, env, positions=[], scale=1.0):
        self.target_positions = positions  # [(env.width - 2, env.height - 2)]
        self.scale = scale
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.env.agent_pos in self.target_positions:
            reward *= self.scale
        return obs, reward, terminated, truncated, info


class Gymnasium2GymWrapper(Wrapper):
    """Wrapper to translate the Gymnasium interface into Gym interface."""

    def __init__(self, env):
        super().__init__(env)

        # Translate action space
        if isinstance(self.env.action_space, gymnasium.spaces.Discrete):
            self.env.action_space = gym.spaces.Discrete(self.env.action_space.n)
        elif isinstance(self.env.action_space, gymnasium.spaces.Box):
            self.env.action_space = gym.spaces.Box(
                high=self.env.action_space.high,
                low=self.env.action_space.low,
                shape=self.env.action_space.shape,
                dtype=self.env.action_space.dtype)

        # Translate observation space
        if isinstance(self.env.observation_space, gymnasium.spaces.Discrete):
            self.env.observation_space = gym.spaces.Discrete(self.env.observation_space.n)
        elif isinstance(self.env.observation_space, gymnasium.spaces.Box):
            self.env.observation_space = gym.spaces.Box(
                high=self.env.observation_space.high,
                low=self.env.observation_space.low,
                shape=self.env.observation_space.shape,
                dtype=self.env.observation_space.dtype)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, max(terminated, truncated), info

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

