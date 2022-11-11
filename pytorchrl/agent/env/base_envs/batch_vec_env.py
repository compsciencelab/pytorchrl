from gym import Env
import numpy as np
from pytorchrl.agent.env.vec_envs.vec_env_base import VecEnvBase
from pytorchrl.agent.env.vec_envs.util import copy_obs_dict, dict_to_obs, obs_space_info


class BatchedEnv(Env):
    """
    VecEnv that runs multiple environments and executes batched steps.
    Recommended to use when num_envs > 1 and step() can be parallelized in GPU.

    It uses numpy vectors for parallel execution.
    """

    def __init__(self, num_envs, observation_space, action_space):
        super(BatchedEnv, self).__init__()
        self.num_envs = num_envs
        self.action_space = action_space
        self.observation_space = observation_space
        self.keys, self.shapes, self.dtypes = obs_space_info(observation_space)
        self.action_shape = (num_envs, *action_space.shape) if action_space.shape != () else (num_envs, -1)
        self.observation_shape = (num_envs, *observation_space.shape) if observation_space.shape != () else (num_envs, -1)

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        obs = {k: np.zeros((self.num_envs,) + tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys}
        return dict_to_obs(obs).reshape(self.observation_shape)

    def reset_single_environment(self, environment_num):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        obs = {k: np.zeros((self.num_envs,) + tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys}
        return dict_to_obs(obs).reshape(self.observation_shape)

    def step(self, action):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        obs = {k: np.zeros((self.num_envs, ) + tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys}
        rews = np.zeros((self.num_envs, ), dtype=np.float32)
        dones = np.ones((self.num_envs, ), dtype=np.bool)
        infos = [{} for _ in range(self.num_envs)]
        return dict_to_obs(obs).reshape(self.observation_shape), rews, dones, infos

