from gym import Env
import numpy as np
from abc import ABC, abstractmethod
from pytorchrl.agent.env.vec_envs.vec_env_base import VecEnvBase
from pytorchrl.agent.env.vec_envs.util import copy_obs_dict, dict_to_obs, obs_space_info


# TODO: are observation_space and action_space required as a parameter?


class BatchedEnv(Env):
    """
    Environment class that runs multiple environments in a single thread.

    Recommended when num_envs > 1 and batching multiple score computations is faster than running individual
    parallel calculations in independent threads (e.g. calculations are parallelized in GPU).

    Observations, actions and done flags have to be numpy arrays. Dict has to be a list of dicts. When one of
    the environments reaches the end of an episode, it is automatically reset. Also the method reset_single_env
    can be called to reset a single environment.
    """

    def __init__(self, num_envs, observation_space=None, action_space=None, **kwargs):
        super(BatchedEnv, self).__init__()
        self.num_envs = num_envs
        # self.action_space = action_space
        # self.observation_space = observation_space
        # self.keys, self.shapes, self.dtypes = obs_space_info(observation_space)
        # self.action_shape = (num_envs, *action_space.shape) if action_space.shape != () else (num_envs, -1)
        # self.observation_shape = (num_envs, *observation_space.shape) if observation_space.shape != () else (num_envs, -1)

    def _check_obs(self, obs):
        """Returns True if obs has the correct type and shape."""
        return isinstance(obs, np.array) and obs.shape[0] == self.num_envs

    def _check_rew(self, rew):
        """Returns True if rew has the correct type and shape."""
        return isinstance(rew, np.array) and rew.shape[0] == self.num_envs

    def _check_done(self, done):
        """Returns True if done has the correct type and shape."""
        return isinstance(done, np.array) and done.shape[0] == self.num_envs

    def _check_info(self, info):
        """Returns True if info has the correct type and shape."""
        return isinstance(info, list) and len(info) == self.num_envs and isinstance(infos[0], dict)

    def reset(self):
        """Reset all the environments and return an array of observations."""
        raise NotImplementedError

    def reset_single_env(self, num_env):
        """Reset environment in position num_env and return an the whole array of observations."""
        raise NotImplementedError

    def step(self, action):
        """
        Takes a step for each environment in the batch of environments. Takes in a np.array of actions and
        returns np.arrays of obs, rews, dones and a list of infos.
        """
        raise NotImplementedError


