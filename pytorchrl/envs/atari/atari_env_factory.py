from pytorchrl.envs.atari.wrappers import wrap_deepmind, make_atari


def atari_train_env_factory(env_id, index_col_worker, index_grad_worker, index_env=0, seed=0, frame_stack=1):
    """
    Create train Atari environment.

    Parameters
    ----------
    env_id : str
        Environment name.
    index_col_worker : int
        Index of the collection worker running this environment.
    index_grad_worker : int
        Index of the gradient worker running the collection worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
    seed : int
        Environment random seed.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.

    Returns
    -------
    env : gym.Env
        Train environment.
    """
    env = make_atari(env_id)
    env.seed(index_grad_worker * 1000 + 100 * index_col_worker + index_env + seed)
    env = wrap_deepmind(
        env, episode_life=True,
        clip_rewards=True,
        scale=False,
        frame_stack=frame_stack)

    return env


def atari_test_env_factory(env_id, index_col_worker, index_grad_worker, index_env=0, seed=0, frame_stack=1):
    """
    Create test Atari environment.

    Parameters
    ----------
    env_id : str
        Environment name.
    index_col_worker : int
        Index of the collection worker running this environment.
    index_grad_worker : int
        Index of the gradient worker running the collection worker running this environment.
    index_env : int
        Index of this environment withing the vector of environments.
    seed : int
        Environment random seed.
    frame_stack : int
        Observations composed of last `frame_stack` frames stacked.

    Returns
    -------
    env : gym.Env
        Test environment.
    """
    env = make_atari(env_id)
    env.seed(index_grad_worker * 1000 + 100 * index_col_worker + index_env + seed)
    env = wrap_deepmind(
        env, episode_life=False,
        clip_rewards=False,
        scale=False,
        frame_stack=frame_stack)

    return env