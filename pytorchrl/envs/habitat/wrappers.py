import gym
import numpy as np
from typing import Optional
import habitat
from habitat import Config, Dataset
from pytorchrl.envs.habitat.utils import center_crop, scale_image


class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self.SUCCESS_REWARD = 2.5
        self.SLACK_REWARD = -0.001
        self.REWARD_MEASURE = 'distance_to_goal'
        self.SUCCESS_MEASURE = 'success'
        self._core_env_config = config
        self._reward_measure_name = self.REWARD_MEASURE
        self._success_measure_name = self.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[self._reward_measure_name]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        obs, reward, done, info = super().step(*args, **kwargs)
        return obs, reward, done, info

    def get_reward_range(self):
        return (
            self.SLACK_REWARD - 1.0,
            self.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


class MyHabitat(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
        self._action_space_dict = self.env.action_space
        self.action_space = gym.spaces.Discrete(4)

    def step(self, action):
        action_env = self._transform_action(action)
        obs, reward, done, info = self.env.step(action=action_env)
        obs = self._transform_obs(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self._transform_obs(obs)
        return obs

    def _transform_obs(self, obs):
        img = obs['rgb']
        # crop and rescale 360,640 to 256,256
        img = center_crop(img, (360, 360))
        # img = scale_image(img, 256.0/360.0)
        img = scale_image(img, 84.0 / 360.0)
        return img

    def _transform_action(self, action_index):
        return {
            "action": list(self._action_space_dict.spaces.keys())[action_index],
            "action_args": list(self._action_space_dict.spaces.values())[action_index].sample(),
        }