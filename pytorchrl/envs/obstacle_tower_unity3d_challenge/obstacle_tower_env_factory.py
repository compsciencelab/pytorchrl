import os
import numpy as np
import obstacle_tower_env
from obstacle_tower_env import ObstacleTowerEnv
from ..common import FrameStack, FrameSkip
from .wrappers import ReducedActionEnv, BasicObstacleEnv, RewardShapeObstacleEnv, BasicObstacleEnvTest


def obstacle_train_env_factory(
        index_worker=0, rank=0, frame_skip=0, frame_stack=1, min_floor=0,
        max_floor=50, reduced_actions=True, reward_shape=True):
    """
    Create train Obstacle Tower Unity3D environment.
    Useful info_keywords 'floor', 'start', 'seed'.

    Parameters
    ----------
    frame_skip : int
        Return only every `frame_skip`-th observation.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.
    min_floor : int
        Minimum floor the agent can be spawned in.
    max_floor : int
        Maximum floor the agent can be spawned in.
    reduced_actions : bool
        Whether or not to use the action wrapper to reduce the number of available actions.
    reward_shape : bool
        Whether or not to use the reward shape wrapper.

    Returns
    -------
    env : gym.Env
        Train environment.
    """
    if 'DISPLAY' not in os.environ.keys():
        os.environ['DISPLAY'] = ':0'

    exe = os.path.join(os.path.dirname(
        obstacle_tower_env.__file__), 'ObstacleTower/obstacletower')

    env = ObstacleTowerEnv(
        environment_filename=exe, retro=True, worker_id=index_worker + rank + np.random.randint(1, 10000),
        greyscale=False, docker_training=False, realtime_mode=False)

    if reduced_actions:
        env = ReducedActionEnv(env)

    env = BasicObstacleEnv(env, max_floor=max_floor, min_floor=min_floor)

    if reward_shape:
        env = RewardShapeObstacleEnv(env)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    return env

def obstacle_test_env_factory(
        index_worker=0, rank=0, frame_skip=0, frame_stack=1, realtime=False,
        min_floor=0, max_floor=50, reduced_actions=True):
    """
    Create test Obstacle Tower Unity3D environment.
    Useful info_keywords 'floor', 'start', 'seed'.

    Parameters
    ----------
    frame_skip : int
        Return only every `frame_skip`-th observation.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.
    min_floor : int
        Minimum floor the agent can be spawned in.
    max_floor : int
        Maximum floor the agent can be spawned in.
    reduced_actions : bool
        Whether or not to use the action wrapper to reduce the number of available actions.

    Returns
    -------
    env : gym.Env
        Train environment.
    """

    if 'DISPLAY' not in os.environ.keys():
        os.environ['DISPLAY'] = ':0'

    exe = os.path.join(os.path.dirname(
        obstacle_tower_env.__file__), 'ObstacleTower/obstacletower')

    env = ObstacleTowerEnv(
        environment_filename=exe, retro=True, worker_id=index_worker + rank + np.random.randint(1, 10000),
        greyscale=False, docker_training=False, realtime_mode=realtime)

    if reduced_actions:
        env = ReducedActionEnv(env)

    env = BasicObstacleEnvTest(env, max_floor=max_floor, min_floor=min_floor)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    return env