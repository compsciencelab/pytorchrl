from gym import Env
import numpy as np
from pytorchrl.agent.env.vec_envs.vec_env_base import VecEnvBase
from pytorchrl.agent.env.vec_envs.util import copy_obs_dict, dict_to_obs, obs_space_info


class BatchedEnv(Env):
    """
    VecEnv that runs multiple environments and executes batched steps.
    Recommended to use when num_envs > 1 and step() can be parallelized in GPU.

    Obs, actions and dones have to be numpy arrays. Create sanity checks

    dict has to be a list of dicts. Create sanity checks.

    Reset is done automatically !!! But reset and reset_single_env can be called as well
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
        Reset all the environments and return an array of observations.
        """
        obs = {k: np.zeros((self.num_envs,) + tuple(self.shapes[k]), dtype=self.dtypes[k]) for k in self.keys}
        return dict_to_obs(obs).reshape(self.observation_shape)

    def reset_single_env(self, num_env):
        """Reset environment in position num_env and return an the whole array of observations."""
        pass

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
        dones = np.zeros((self.num_envs, ), dtype=np.bool)
        infos = [{} for _ in range(self.num_envs)]
        return dict_to_obs(obs).reshape(self.observation_shape), rews, dones, infos

