import os
import numpy as np
import obstacle_tower_env
from obstacle_tower_env import ObstacleTowerEnv
from pytorchrl.envs.common import FrameStack, FrameSkip
from pytorchrl.envs.obstacle_tower.wrappers import (
    ReducedActionEnv, BasicObstacleEnv, RewardShapeObstacleEnv, BasicObstacleEnvTest)


def obstacle_train_env_factory(
        index_col_worker, index_grad_worker, index_env=0, frame_skip=0, frame_stack=1, min_floor=0,
        max_floor=50, reduced_actions=True, reward_shape=True, exe_path=None):
    """
    Create train Obstacle Tower Unity3D environment.
    Useful info_keywords 'floor', 'start', 'seed'.

    Parameters
    ----------
     index_col_worker : int
        Index of the collection worker running this environment.
    index_grad_worker : int
        Index of the gradient worker running the collection worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
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
    exe_path : str
        Path to obstacle environment executable.

    Returns
    -------
    env : gym.Env
        Train environment.
    """
    if 'DISPLAY' not in os.environ.keys():
        os.environ['DISPLAY'] = ':0'

    if exe_path:
        exe = exe_path
    else:
        exe = os.path.join(os.path.dirname(
            obstacle_tower_env.__file__), 'ObstacleTower/obstacletower')

    id = index_grad_worker * 1000 + 100 * index_col_worker + index_env
    env = ObstacleTowerEnv(
        environment_filename=exe, retro=True, worker_id=id,
        greyscale=False, timeout_wait=60, realtime_mode=False)

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
        index_col_worker, index_grad_worker, index_env=0, frame_skip=0, frame_stack=1, realtime=False,
        min_floor=0, max_floor=50, reduced_actions=True, exe_path=None):
    """
    Create test Obstacle Tower Unity3D environment.
    Useful info_keywords 'floor', 'start', 'seed'.

    Parameters
    ----------
    index_col_worker : int
        Index of the collection worker running this environment.
    index_grad_worker : int
        Index of the gradient worker running the collection worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
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
    exe_path : str
        Path to obstacle environment executable.

    Returns
    -------
    env : gym.Env
        Train environment.
    """

    if 'DISPLAY' not in os.environ.keys():
        os.environ['DISPLAY'] = ':0'

    if exe_path:
        exe = exe_path
    else:
        exe = os.path.join(os.path.dirname(
            obstacle_tower_env.__file__), 'ObstacleTower/obstacletower')

    id = index_grad_worker * 1000 + 100 * index_col_worker + index_env
    env = ObstacleTowerEnv(
        environment_filename=exe, retro=True, worker_id=id,
        greyscale=False, realtime_mode=realtime)

    if reduced_actions:
        env = ReducedActionEnv(env)

    env = BasicObstacleEnvTest(env, max_floor=max_floor, min_floor=min_floor)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    return env