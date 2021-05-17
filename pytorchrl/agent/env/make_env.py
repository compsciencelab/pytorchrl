import os
import inspect
from pytorchrl.agent.env.openai_baselines_dependencies.Monitor import Monitor

from pytorchrl.agent.env.env_wrappers import TransposeImagesIfRequired


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
            env = Monitor(
                env, os.path.join(path, "{}_{}_{}".format(
                    index_grad_worker, index_col_worker, str(index_env))),
                allow_early_resets=True, info_keywords=info_keywords)

        # if obs are images with shape (W,H,C), transpose to (C,W,H) for PyTorch convolutions
        env = TransposeImagesIfRequired(env, op=[2, 0, 1])

        return env

    return _thunk