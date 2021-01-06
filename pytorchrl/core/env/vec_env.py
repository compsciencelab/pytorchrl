import os
import torch
import inspect
from baselines import bench
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from .vector_wrappers import VecPyTorch
from .env_wrappers import TransposeImage


class VecEnv:
    """Class to handle creation of environment vectors"""

    @classmethod
    def create_factory(cls, env_fn, env_kwargs={}, vec_env_size=1, log_dir=None, info_keywords=()):
        """
        Returns a function to create a vector of environments of size
        num_processes, so it can be executed by any worker, remote or not.

        Parameters
        ----------
        env_fn : func
            Function to create the environment.
        env_kwargs : dict
            keyword arguments of env_fn.
        vec_env_size : int
            size of the vector of environments.
        log_dir : str
            Target path for envs to log information through bench.Monitor class.
        info_keywords : tuple
            Information keywords to be logged stored by bench.Monitor class.

        Returns
        -------
        make_vec_env : func
            Function to create a vector of environments.
        dummy_env.action_space : gym.Space
            Environments action space.
        dummy_env.observation_space: gym.Space
            Environments observation space.
        """

        def make_vec_env(device=torch.device("cpu"), index_col_worker=1, index_grad_worker=1, mode="train"):
            """Create and return a vector environment"""

            if mode == "train":
                env_indexes = range(0, vec_env_size)
            else:
                env_indexes = range(vec_env_size, 2 * vec_env_size)

            envs = [make_env(
                env_fn=env_fn, env_kwargs=env_kwargs,
                index_col_worker=index_col_worker,
                index_grad_worker=index_grad_worker,
                index_env=i, log_dir=log_dir, mode=mode,
                info_keywords=info_keywords
            ) for i in env_indexes]

            if len(envs) > 1:
                envs = SubprocVecEnv(envs)
            else:
                envs = DummyVecEnv(envs)

            envs = VecPyTorch(envs, device)

            return envs

        dummy_env = [make_env(
            env_fn=env_fn, env_kwargs=env_kwargs, index_col_worker=0,
            index_grad_worker=0, index_env=0)]
        dummy_env = DummyVecEnv(dummy_env)

        cls.action_space = dummy_env.action_space
        cls.observation_space = dummy_env.observation_space

        return make_vec_env, dummy_env.action_space, dummy_env.observation_space

def make_env(env_fn, env_kwargs, index_col_worker, index_grad_worker, index_env, log_dir=None, info_keywords=(), mode="train"):
    """
    Returns a function that handles the creating of a single environment, so it
    can be executed in an independent thread.

    Parameters
    ----------

    env_fn : func
        Function to create the environment.
    env_kwargs : dict
        keyword arguments of env_fn.
    log_dir : str
        Target path for bench.Monitor logger values.
    info_keywords : tuple
        Information keywords to be logged stored by bench.Monitor.
    index_col_worker:
        Index of the data collection worker running this environment.
    index_grad_worker : int
        Index of the gradient worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
    mode : str
        "train" or "test"

    Returns
    -------
    _thunk : func
        A function to create and return the environment. It also sets up the
        Monitor logging and used a TransposeImage wrapper if environment obs
        are images.

    """

    if log_dir:
        path = os.path.join(log_dir, mode)
        os.makedirs(path, exist_ok=True)

    # index_worker and index_env added as paramaters
    if "index_col_worker" in inspect.getfullargspec(env_fn).args:
        env_kwargs["index_col_worker"] = index_col_worker
    if "index_grad_worker" in inspect.getfullargspec(env_fn).args:
        env_kwargs["index_grad_worker"] = index_grad_worker
    if "index_env" in inspect.getfullargspec(env_fn).args:
        env_kwargs["index_env"] = index_env

    def _thunk():
        """Creates and returns environment"""

        # Create single environment
        env = env_fn(**env_kwargs)

        # Monitor provided info_keywords
        if log_dir is not None:
            env = bench.Monitor(
                env, os.path.join(path, "{}_{}_{}".format(
                    index_grad_worker, index_col_worker, str(index_env))),
                allow_early_resets=True, info_keywords=info_keywords)

        # if obs are images with shape (W,H,C), transpose to (C,W,H) for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk
