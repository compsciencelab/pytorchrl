import os
from pytorchrl.envs.generative_chemistry.rnn_reinvent_environment import GenChemEnv
# from pytorchrl.envs.generative_chemistry.transformer_reinvent_environment import GenChemEnv
# from pytorchrl.envs.generative_chemistry.rnn_libinvent_environment import GenChemEnv
from pytorchrl.envs.generative_chemistry.vocabulary import SMILESTokenizer, create_vocabulary


def generative_chemistry_train_env_factory(scoring_function, vocabulary, smiles_max_length=200,  scaffolds=[]):
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
        List of LibInvent scaffolds

    Returns
    -------
    env : gym.Env
        Train environment.
    """

    env = GenChemEnv(
        scoring_function=scoring_function,
        vocabulary=vocabulary,
        max_length=smiles_max_length)

    return env
