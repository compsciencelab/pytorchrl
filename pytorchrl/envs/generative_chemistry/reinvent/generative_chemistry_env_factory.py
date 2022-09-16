import os
from pytorchrl.envs.generative_chemistry.rnn_reinvent_environment import GenChemEnv as RnnReinvent
from pytorchrl.envs.generative_chemistry.transformer_reinvent_environment import GenChemEnv as TrReinvent
from pytorchrl.envs.generative_chemistry.rnn_libinvent_environment import GenChemEnv as RnnLibInvent
from pytorchrl.envs.generative_chemistry.vocabulary import SMILESTokenizer, create_vocabulary,\
    ReinventVocabulary, LibinventVocabulary


def generative_chemistry_train_env_factory(
        scoring_function, vocabulary, smiles_max_length=200,  scaffolds=[], accumulate_obs=False):
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
    accumulate_obs : bool
        If True, obs are accumulated at every time step (e.g. C, CC, CCC, ...)

    Returns
    -------
    env : gym.Env
        Train environment.
    """

    if isinstance(vocabulary, ReinventVocabulary):
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
    elif isinstance(vocabulary, LibinventVocabulary):
        assert len(scaffolds) > 0, "LibInvent environment requires at least 1 scaffold!"
        env = RnnLibInvent(
            scoring_function=scoring_function,
            vocabulary=vocabulary,
            max_length=smiles_max_length,
            scaffolds=scaffolds)
    else:
        raise ValueError("Vocabulary class not recognised!")

    return env
