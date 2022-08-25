from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task
from pytorchrl.envs.common import FrameStack, FrameSkip, DelayedReward


def causal_world_train_env_factory(task_id="general",
                                   frame_skip=0,
                                   frame_stack=1,
                                   inference=False,
                                   reward_delay=1):
    """
    Create train Animal Olympics Unity3D environment.

    Parameters
    ----------
    task_id : int
        target causal world task to train the agent.
    frame_skip : int
        Return only every `frame_skip`-th observation.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.
    reward_delay : int
        Only return accumulated reward every `reward_delay` steps to simulate sparse reward environment.
    inference : bool
        Whether or not to render the environment in real time.

    Returns
    -------
    env : gym.Env
        Train environment.
    """
    task = generate_task(task_generator_id=task_id)
    env = CausalWorld(task=task, enable_visualization=inference)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    if reward_delay > 1:
        env = DelayedReward(env, delay=reward_delay)

    return env
