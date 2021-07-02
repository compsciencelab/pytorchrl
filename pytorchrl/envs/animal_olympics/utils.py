import os
import gym
import glob
import random
import numpy as np
from PIL import Image

from animalai.envs.arena_config import ArenaConfig
from animalai.envs.gym.environment import ActionFlattener
from envs.animal_olympics.utils import set_reward_arena


class RetroEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.flattener = ActionFlattener([3, 3])
        self.action_space = self.flattener.action_space
        self.observation_space = gym.spaces.Box(0, 255, dtype=np.uint8, shape=(84, 84, 3))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)  # non-retro
        visual_obs, vector_obs = self._preprocess_obs(obs)
        info['vector_obs'] = vector_obs
        return visual_obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        visual_obs, _ = self._preprocess_obs(obs)
        return visual_obs

    def _preprocess_obs(self, obs):
        visual_obs, vector_obs = obs
        visual_obs = self._preprocess_single(visual_obs)
        visual_obs = self._resize_observation(visual_obs)
        return visual_obs, vector_obs

    @staticmethod
    def _preprocess_single(single_visual_obs):
        return (255.0 * single_visual_obs).astype(np.uint8)

    @staticmethod
    def _resize_observation(observation):
        """
        Re-sizes visual observation to 84x84
        """
        obs_image = Image.fromarray(observation)
        obs_image = obs_image.resize((84, 84), Image.NEAREST)
        return np.array(obs_image)


# {0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [1, 0], 4: [1, 1], 5: [1, 2], 6: [2, 0], 7: [2, 1], 8: [2, 2]}
class FilterActionEnv(gym.ActionWrapper):
    """
    An environment wrapper that limits the action space.
    """
    _ACTIONS = (0, 1, 2, 3, 4, 5, 6)

    def __init__(self, env):
        super().__init__(env)
        self.actions = self._ACTIONS
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def step(self, action):
        action = int(action)
        action = self.flattener.lookup_action(action)  # convert to multi
        obs, reward, done, info = self.env.step(action)  # non-retro
        return obs, reward, done, info

    def action(self, act):
        return self.actions[act]


class LabAnimal(gym.Wrapper):
    def __init__(self, env, arenas_dir):
        gym.Wrapper.__init__(self, env)
        if os.path.isdir(arenas_dir):
            files = glob.glob("{}/*.yaml".format(arenas_dir))
        else:
            # assume is a pattern
            files = glob.glob(arenas_dir)

        self.env_list = [(f, ArenaConfig(f)) for f in files]
        self._arena_file = ''

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.steps += 1
        self.env_reward += reward
        info['arena'] = self._arena_file  # for monitor
        info['max_reward'] = self.max_reward
        info['max_time'] = self.max_time
        info['ereward'] = self.env_reward
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.steps = 0
        self.env_reward = 0
        self._arena_file, arena = random.choice(self.env_list)
        #        self.max_reward = analyze_arena(arena)
        self.max_reward = set_reward_arena(arena, force_new_size=False)
        self.max_time = arena.arenas[0].t
        return self.env.reset(arenas_configurations=arena, **kwargs)


class RewardShaping(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward > -0.005 and reward < 0:  # remove time negative reward
            reward = 0
        if done:  # give time penalty at the end
            reward -= self.steps / self.max_time
        if reward > 0 and done and self.steps < 60:  # explore first
            reward = 0
        if reward > 0 and not done:  # brown ball, go for it first
            reward += 3
        if reward > 0 and self.env_reward > self.max_reward - 1 and done:  # prize for finishing well
            reward += 10
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)