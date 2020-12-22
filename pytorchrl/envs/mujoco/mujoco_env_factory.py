import gym
from ..common import FrameStack, FrameSkip


def mujoco_train_env_factory(env_id, index_worker=0, index_env=0, seed=0, frame_skip=0, frame_stack=1):
    """
    Create train MuJoCo environment.

    Parameters
    ----------
    env_id : str
        Environment name.
    index_worker : int
        Index of the worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
    seed : int
        Environment random seed.
    frame_skip : int
        Return only every `frame_skip`-th observation.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.

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

def mujoco_test_env_factory(env_id, index_worker=0, index_env=0, seed=0, frame_skip=0, frame_stack=1):
    """
    Create test MuJoCo environment.

    Parameters
    ----------
    env_id : str
        Environment name.
    index_worker : int
        Index of the worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
    seed : int
        Environment random seed.
    frame_skip : int
        Return only every `frame_skip`-th observation.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.

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