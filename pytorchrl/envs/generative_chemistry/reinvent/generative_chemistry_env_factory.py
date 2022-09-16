import os
from pytorchrl.envs.generative_chemistry.reinvent.rnn_environment import GenChemEnv as RnnReinvent
from pytorchrl.envs.generative_chemistry.reinvent.transformer_environment import GenChemEnv as TrReinvent
from pytorchrl.envs.generative_chemistry.vocabulary import SMILESTokenizer, create_vocabulary


def reinvent_train_env_factory(scoring_function, vocabulary, smiles_max_length=200, accumulate_obs=False):
    """
    Create train REINVENT environment.

    Parameters
    ----------
    scoring_function : func
        Function that given a SMILE, returns a score.
    vocabulary : class
        Class that stores the tokens and allows their conversion to vocabulary indexes.
    smiles_max_length : int
        Maximum length allowed for the generated SMILES. Equivalent to max episode length.
    accumulate_obs : bool
        If True, obs are accumulated at every time step (e.g. C, CC, CCC, ...)

    Returns
    -------
    env : gym.Env
        Train environment.
    """

    if accumulate_obs:
        env = TrReinvent(
            scoring_function=scoring_function,
            vocabulary=vocabulary,
            max_length=smiles_max_length)
    else:
        env = RnnReinvent(
            scoring_function=scoring_function,
            vocabulary=vocabulary,
            max_length=smiles_max_length)

    return env
