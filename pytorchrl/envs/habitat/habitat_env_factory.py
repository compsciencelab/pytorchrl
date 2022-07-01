import os
import gym
import habitat
from pytorchrl.envs.common import FrameSkip, FrameStack
from pytorchrl.envs.habitat.wrappers import NavRLEnv, MyHabitat


def habitat_env_factory(conf_file, index_worker=0, index_env=0, frame_skip=0, frame_stack=1):
    """
    Create train Obstacle Tower Unity3D environment.
    Useful info_keywords 'floor', 'start', 'seed'.
    Parameters
    ----------
    frame_skip : int
        Return only every `frame_skip`-th observation.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.
    Returns
    -------
    env : gym.Env
        Train environment.
    """
    if 'DISPLAY' not in os.environ.keys():
        os.environ['DISPLAY'] = ':0'

    config = habitat.get_config(conf_file)
    env = NavRLEnv(config)
    env.seed(config['SEED'] + index_worker + index_env)

    env = MyHabitat(env)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    return env
