import os
import copy
import inspect
import torch
from pytorchrl.agent.env.base_envs.batched_env import BatchedEnv
from pytorchrl.agent.env.base_envs.env_wrappers import (
    Monitor,
    PyTorchEnv,
    BatchedMonitor,
    TransposeImagesIfRequired,
)
from pytorchrl.agent.env.vec_envs.vector_wrappers import VecPyTorch
from pytorchrl.agent.env.vec_envs.parallel_vec_env import ParallelVecEnv
from pytorchrl.agent.env.vec_envs.sequential_vec_env import SequentialVecEnv


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
            size of the vector of environments. If env_fn creates an instance BatchedEnv class this
            parameter will be ignored.
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

        if "index_col_worker" in inspect.getfullargspec(env_fn).args:
            env_kwargs["index_col_worker"] = 0
        if "index_grad_worker" in inspect.getfullargspec(env_fn).args:
            env_kwargs["index_grad_worker"] = 0
        if "index_env" in inspect.getfullargspec(env_fn).args:
            env_kwargs["index_env"] = 0
        dummy_env = env_fn(**env_kwargs)

        if isinstance(dummy_env, BatchedEnv):

            dummy_env = TransposeImagesIfRequired(dummy_env, op=[2, 0, 1])
            cls.action_space = dummy_env.action_space
            cls.observation_space = dummy_env.observation_space
            dummy_env.close()

            def make_vec_env(device=torch.device("cpu"), index_col_worker=1, index_grad_worker=1, mode="train"):
                """Create and return a vector environment"""

                if "index_col_worker" in inspect.getfullargspec(env_fn).args:
                    env_kwargs["index_col_worker"] = index_col_worker
                if "index_grad_worker" in inspect.getfullargspec(env_fn).args:
                    env_kwargs["index_grad_worker"] = index_grad_worker
                if "mode" in inspect.getfullargspec(env_fn).args:
                    env_kwargs["mode"] = mode

                env = env_fn(**env_kwargs)

                if log_dir:
                    path = os.path.join(log_dir, "monitor_logs", mode)
                    os.makedirs(path, exist_ok=True)
                    env = BatchedMonitor(
                        env, os.path.join(path, "{}_{}".format(
                            index_grad_worker, index_col_worker)),
                        info_keywords=info_keywords)

                # if obs are images with shape (W,H,C), transpose to (C,W,H) for PyTorch convolutions
                env = TransposeImagesIfRequired(env, op=[2, 0, 1])

                env = PyTorchEnv(env, device)

                env.env_kwargs = env_kwargs

                return env

        else:

            dummy_env.close()
            dummy_env = [make_env(
                env_fn=env_fn, env_kwargs=env_kwargs, index_col_worker=0,
                index_grad_worker=0, index_env=0)]
            dummy_env = SequentialVecEnv(dummy_env)
            cls.action_space = dummy_env.action_space
            cls.observation_space = dummy_env.observation_space
            dummy_env.envs[0].close()

            def make_vec_env(device=torch.device("cpu"), index_col_worker=1, index_grad_worker=1, mode="train"):
                """Create and return a vector environment"""

                if mode == "train":
                    env_indexes = range(1, vec_env_size + 1)
                else:
                    env_indexes = range(1 + vec_env_size, 2 * vec_env_size + 1)

                envs = [make_env(
                    env_fn=env_fn, env_kwargs=env_kwargs,
                    index_col_worker=index_col_worker,
                    index_grad_worker=index_grad_worker,
                    index_env=i, log_dir=log_dir, mode=mode,
                    info_keywords=info_keywords
                ) for i in env_indexes]

                if len(envs) > 1:
                    envs = ParallelVecEnv(envs)
                else:
                    envs = SequentialVecEnv(envs)

                envs = VecPyTorch(envs, device)

                envs.env_kwargs = env_kwargs

                return envs

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
        path = os.path.join(log_dir, "monitor_logs", mode)
        os.makedirs(path, exist_ok=True)

    # index_worker and index_env added as paramaters
    env_kwargs = copy.deepcopy(env_kwargs)
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
            env = Monitor(
                env, os.path.join(path, "{}_{}_{}".format(
                    index_grad_worker, index_col_worker, str(index_env))),
                info_keywords=info_keywords)

        # if obs are images with shape (W,H,C), transpose to (C,W,H) for PyTorch convolutions
        env = TransposeImagesIfRequired(env, op=[2, 0, 1])

        return env

    return _thunk
