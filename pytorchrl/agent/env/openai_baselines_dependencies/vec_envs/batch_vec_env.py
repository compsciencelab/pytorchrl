import gym
import numpy as np
from pytorchrl.agent.env.openai_baselines_dependencies.vec_envs.vec_env_base import VecEnvBase
from pytorchrl.agent.env.openai_baselines_dependencies.vec_envs.util import copy_obs_dict, dict_to_obs, obs_space_info

# Steps
#    1. VecEnv create factory could allow to create a batched environment
#    2. BatchedVecEnv would be just a type of environment, maybe needs to inherit from gym.env?
#    3.
#    4.
#    5.


class BatchedVecEnv(VecEnvBase):
    """
    VecEnv that runs multiple environments and executes batched steps.
    Recommended to use when num_envs > 1 and step() can be parallelized in GPU.

    It uses numpy vectors for parallel execution.
    """

    def __init__(self, num_envs, observation_space, action_space):
        VecEnvBase.__init__(self, num_envs, observation_space, action_space)
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

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
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

