import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, ReseedWrapper
from pytorchrl.envs.minigrid.custom_environments import register_custom_minigrid_envs
from pytorchrl.envs.minigrid.wrappers import RewardShapeWrapper, Gymnasium2GymWrapper

# Register custom environments
register_custom_minigrid_envs()


def minigrid_train_env_factory(env_id, index_worker=0, index_env=0, seed=None):
    """
    Create train PyBullet environment.

    Parameters
    ----------
    env_id : str
        Environment name.
    index_worker : int
        Index of the worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.

    Returns
    -------
    env : gym.Env
        Train environment.
    """

    env = gym.make(env_id, tile_size=32)

    env = RewardShapeWrapper(env)

    if seed:
        # Fix environment seed
        env = ReseedWrapper(env, seeds=[seed])

    # Get pixel observations
    env = RGBImgPartialObsWrapper(env, tile_size=8)

    # Get rid of the 'mission' field
    env = ImgObsWrapper(env)

    env = Gymnasium2GymWrapper(env)

    return env
