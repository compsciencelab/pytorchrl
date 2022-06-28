import gym
import crafter
from pytorchrl.envs.common import FrameStack, FrameSkip


def crafter_train_env_factory(env_id, index_worker=0, index_env=0, seed=0, frame_skip=0, frame_stack=1):
    """
    Create train Crafter environment.

    Parameters
    ----------
    env_id : str
        Environment name.
    index_worker : int
        Index of the collection worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
    seed : int
        Environment random seed.


    Returns
    -------
    env : gym.Env
        Train environment.
    """
    env = gym.make(env_id)
    env.seed(seed + index_worker + index_env)
    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    return env


def crafter_test_env_factory(env_id, index_worker=0, index_env=0, seed=0, frame_skip=0, frame_stack=1):
    """
    Create test Crafter environment.

    Parameters
    ----------
    env_id : str
        Environment name.
    index_worker : int
        Index of the collection worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
    seed : int
        Environment random seed.


    Returns
    -------
    env : gym.Env
        Test environment.
    """
    env = gym.make(env_id)
    env.seed(seed + index_worker + index_env)
    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    return env


