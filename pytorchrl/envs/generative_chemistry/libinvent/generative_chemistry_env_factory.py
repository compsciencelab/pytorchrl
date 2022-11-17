import os
from pytorchrl.envs.generative_chemistry.libinvent.rnn_environment import GenChemEnv as RnnLibInvent
from pytorchrl.envs.generative_chemistry.libinvent.batched_rnn_environment import BatchedGenChemEnv as BatchedRnnLibInvent
from pytorchrl.envs.generative_chemistry.vocabulary import SMILESTokenizer, create_vocabulary


def libinvent_train_env_factory(
        scoring_function, vocabulary, smiles_max_length=200,  scaffolds=[],
        randomize_scaffolds=False, reaction_filters=[], accumulate_obs=False):
    """
    Create train GenChem environment.

    Parameters
    ----------
    scoring_function : func
        Function that given a SMILE, returns a score.
    vocabulary : class
        Class that stores the tokens and allows their conversion to vocabulary indexes.
    smiles_max_length : int
        Maximum length allowed for the generated SMILES. Equivalent to max episode length.
    scaffolds : list
        List of LibInvent scaffolds to be decorated.
    randomize_scaffolds : bool
        Whether or not a random SMILES representation of the scaffolds should be used at each episode.
        Crucially, this is not yet possible if a selective reaction filter is imposed.
    reaction_filters : list
        Reaction filters favored in the reward function.
    accumulate_obs : bool
        If True, obs are accumulated at every time step (e.g. C, CC, CCC, ...)

    Returns
    -------
    env : gym.Env
        Train environment.
    """

    env = RnnLibInvent(
        scoring_function=scoring_function,
        vocabulary=vocabulary,
        max_length=smiles_max_length,
        scaffolds=scaffolds,
        randomize_scaffolds=randomize_scaffolds,
        reactions=reaction_filters,
    )

    return env


def libinvent_train_batched_env_factory(
        scoring_function, vocabulary, smiles_max_length=200,  scaffolds=[],
        randomize_scaffolds=False, reaction_filters=[], num_envs=10):
    """
    Create train GenChem environment.

    Parameters
    ----------
    scoring_function : func
        Function that given a SMILE, returns a score.
    vocabulary : class
        Class that stores the tokens and allows their conversion to vocabulary indexes.
    smiles_max_length : int
        Maximum length allowed for the generated SMILES. Equivalent to max episode length.
    scaffolds : list
        List of LibInvent scaffolds to be decorated.
    randomize_scaffolds : bool
        Whether or not a random SMILES representation of the scaffolds should be used at each episode.
        Crucially, this is not yet possible if a selective reaction filter is imposed.
    reaction_filters : list
        Reaction filters favored in the reward function.
    accumulate_obs : bool
        If True, obs are accumulated at every time step (e.g. C, CC, CCC, ...)

    Returns
    -------
    env : gym.Env
        Train environment.
    """

    env = BatchedRnnLibInvent(
        scoring_function=scoring_function,
        vocabulary=vocabulary,
        max_length=smiles_max_length,
        scaffolds=scaffolds,
        randomize_scaffolds=randomize_scaffolds,
        reactions=reaction_filters,
        num_envs=num_envs,
    )

    return env
