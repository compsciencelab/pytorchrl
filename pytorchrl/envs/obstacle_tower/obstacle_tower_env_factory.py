import os
import obstacle_tower_env
from obstacle_tower_env import ObstacleTowerEnv
from pytorchrl.envs.common import FrameStack, FrameSkip, DelayedReward
from pytorchrl.envs.obstacle_tower.wrappers import (
    ReducedActionEnv, BasicObstacleEnv, RewardShapeObstacleEnv, BasicObstacleEnvTest)


def obstacle_train_env_factory(
        index_col_worker, index_grad_worker, index_env=0, frame_skip=0, frame_stack=1, min_floor=0,
        max_floor=50, reduced_actions=True, num_actions=6, reward_shape=True, exe_path=None, reward_delay=1,
        realtime=False, seed_list=[], id_offset=1, timeout_wait=180):
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
    num_actions : int
        Size of the reduced action space.
    reward_shape : bool
        Whether or not to use the reward shape wrapper.
    exe_path : str
        Path to obstacle environment executable.
    reward_delay : int
        Only return accumulated reward every `reward_delay` steps to simulate sparse reward environment.
    realtime : bool
        Whether or not to render the environment frames in real time.
    seed_list : list
        List of environment seeds to use.
    id_offset : int
        offset added to worker_id to avoid collisions with other runs in the same machine.
    timeout_wait : int
        Time for python interface to wait for environment to connect (in seconds).

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

    id = id_offset + index_grad_worker * 1000 + 100 * index_col_worker + index_env
    env = ObstacleTowerEnv(
        environment_filename=exe, retro=True, worker_id=id,
        greyscale=False, timeout_wait=timeout_wait, realtime_mode=realtime)

    if reduced_actions:
        env = ReducedActionEnv(env, num_actions=num_actions)

    env = BasicObstacleEnv(env, min_floor=min_floor, max_floor=max_floor, seed_list=seed_list)

    if reward_shape:
        env = RewardShapeObstacleEnv(env)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    if reward_delay > 1:
        env = DelayedReward(env, delay=reward_delay)

    return env


def obstacle_test_env_factory(
        index_col_worker, index_grad_worker, index_env=0, frame_skip=0, frame_stack=1, realtime=False,
        min_floor=0, max_floor=50, reduced_actions=True, num_actions=6, exe_path=None, reward_delay=1,
        id_offset=1, timeout_wait=180):
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
    num_actions : int
        Size of the reduced action space.
    exe_path : str
        Path to obstacle environment executable.
    reward_delay : int
        Only return accumulated reward every `reward_delay` steps to simulate sparse reward environment.
    realtime : bool
        Whether or not to render the environment frames in real time.
    id_offset : int
        offset added to worker_id to avoid collisions with other runs in the same machine.
    timeout_wait : int
        Time for python interface to wait for environment to connect (in seconds).

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

    id = id_offset + index_grad_worker * 1000 + 100 * index_col_worker + index_env
    env = ObstacleTowerEnv(
        environment_filename=exe, retro=True, worker_id=id,
        greyscale=False, timeout_wait=timeout_wait, realtime_mode=realtime)

    if reduced_actions:
        env = ReducedActionEnv(env, num_actions=num_actions)

    env = BasicObstacleEnvTest(env, max_floor=max_floor, min_floor=min_floor)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    if reward_delay > 1:
        env = DelayedReward(env, delay=reward_delay)

    return env
