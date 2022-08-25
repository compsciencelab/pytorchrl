import os
from pytorchrl.envs.generative_chemistry.environment import GenChemEnv
from pytorchrl.envs.generative_chemistry.vocabulary import SMILESTokenizer, create_vocabulary


def generative_chemistry_train_env_factory(
        scoring_function, tokenizer=None, vocabulary=None, smiles_max_length=200, smiles_list=[]):
    """
    Create train GenChem environment.

    Parameters
    ----------
    scoring_function : func
        Function that given a SMILE, returns a score.
    tokenizer : class
        Class that deals with the tokenization and untokenization of SMILES.
    vocabulary : class
        Class that stores the tokens and allows their conversion to vocabulary indexes.
    smiles_max_length : int
        Maximum length allowed for the generated SMILES. Equivalent to max episode length.
    smiles_list : list
        List of smiles from which to create a vocabulary. Only used if tokenizer and vocabulary are None.

    Returns
    -------
    env : gym.Env
        Train environment.
    """

    if tokenizer is None and vocabulary is None:
        tokenizer = SMILESTokenizer()
        vocabulary = create_vocabulary(smiles_list, tokenizer=tokenizer)

    env = GenChemEnv(
        scoring_function=scoring_function,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        max_length=smiles_max_length)

    return env


if __name__ == "__main__":

    from pytorchrl.envs.generative_chemistry.default_scoring_function import scoring_function

    env = generative_chemistry_train_env_factory(scoring_function, smiles_list=["[*:0]N1CCN(CC1)CCCCN[*:1]"])
    print("\nSummary:\n")
    print(f"observation_space {env.observation_space}")
    print(f"action_space {env.action_space}")
    obs = env.reset()
    obs2, rew, info, done = env.step("[*:0]")
    obs2, rew, info, done = env.step("N")
    obs2, rew, info, done = env.step("1")
    obs2, rew, info, done = env.step("C")
    obs2, rew, info, done = env.step("C")
    obs2, rew, info, done = env.step("N")
    obs2, rew, info, done = env.step("(")
    obs2, rew, info, done = env.step("C")
    obs2, rew, info, done = env.step("C")
    obs2, rew, info, done = env.step("1")
    obs2, rew, info, done = env.step(")")
    obs2, rew, info, done = env.step("C")
    obs2, rew, info, done = env.step("C")
    obs2, rew, info, done = env.step("C")
    obs2, rew, info, done = env.step("C")
    obs2, rew, info, done = env.step("N")
    obs2, rew, info, done = env.step("[*:1]")
    obs2, rew, info, done = env.step("$")
    env.render()

    print("\nSuccess!\n")
