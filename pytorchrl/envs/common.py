import gym
from gym import spaces
from collections import deque
import numpy as np


class CartPoleActionWrapper(gym.Wrapper):
    def __init__(self, env):
        """CartPole expects scalar integer inputs and not numpy array """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=1):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        last_frame = obs
        return last_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """
        Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        # return LazyFrames(list(self.frames))
        return np.concatenate(self.frames, axis=-1)


class DelayedReward(gym.Wrapper):
    def __init__(self, env, delay=1):
        """
        Returns accumulated non-zero reward only every `delay`-th steps.
        Can be used to simulate sparse-rewards environments.
        """
        gym.Wrapper.__init__(self, env)
        self._delay = delay
        self._step = 0
        self._reward = 0.0

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""

        self._step += 1
        obs, reward, done, info = self.env.step(action)
        self._reward += reward

        if self._step % self._delay == 0 or done:
            reward = self._reward
            self._reward = 0.0
        else:
            reward = 0.0

        return obs, reward, done, info

    def reset(self, **kwargs):
        self._step = 0
        self._reward = 0.0
        return self.env.reset(**kwargs)


class SparseReward(gym.Wrapper):
    def __init__(self, env, threshold=100):
        """
        Returns accumulated non-zero reward only every `delay`-th steps.
        Can be used to simulate sparse-rewards environments.
        """
        gym.Wrapper.__init__(self, env)
        self._threshold = threshold
        self._reward = 0.0

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""

        obs, reward, done, info = self.env.step(action)
        self._reward += reward

        if self._reward // self._threshold > 0 or done:
            reward = (self._reward // self._threshold) * self._threshold
            self._reward = self._reward % self._threshold
        else:
            reward = 0.0

        return obs, reward, done, info

    def reset(self, **kwargs):
        self._reward = 0.0
        return self.env.reset(**kwargs)
