import gym
import numpy as np
from gym import spaces

from pytorchrl.envs.generative_chemistry.string_space import Char


# TODO: review if smiles have to start and end with special characters!


class GenChemEnv(gym.Env):
    """Custom Environment for Generative Chemistry RL."""

    metadata = {'render.modes': ['human']}

    def __init__(self, scoring_function, scaffold, vocabulary, tokenizer, obs_length=50, **kwargs):
        super(GenChemEnv, self).__init__()

        self.num_episodes = 0
        self.tokenizer = tokenizer
        self.obs_length = obs_length
        self.vocabulary = vocabulary
        self.scaffold = scaffold
        self.scoring_function = scoring_function

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = Char(vocab=vocabulary.tokens(), max_length=obs_length)
        self.observation_space = Char(vocab=vocabulary.tokens(), max_length=obs_length)

        # Make sure scaffold is a string
        assert isinstance(scaffold, str), "Base molecule is not a string"

    def step(self, action):
        """Execute one time step within the environment"""

        if not isinstance(action, str):
            action = self.vocabulary.decode(action.squeeze().tolist())
            action = self.tokenizer.untokenize(action)

        self.new_molecule = action
        try:
            reward = self._scoring(self.new_molecule)
        except TypeError:
            reward = 0.0  # Invalid molecule

        info = {}
        done = True
        new_obs = np.zeros(1)  # Does not matter

        return new_obs, reward, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.num_episodes += 1
        tokenized_scaffold = self.tokenizer.tokenize(self.scaffold)
        tokenized_scaffold = ["^"] + tokenized_scaffold  # Start token
        tokenized_scaffold += ["$"] * (self.obs_length - len(tokenized_scaffold))  # End token
        return self.vocabulary.encode(tokenized_scaffold)

    def render(self, mode='human'):
        """Render the environment to the screen"""

        print(f'Scaffold: {self.scaffold}')
        print(f'Decorated Scaffold: {self.new_molecule}')
        print(f'Vocabulary: {self.vocabulary._tokens}')

    def _scoring(self, smiles):
        """Return scoring metric."""

        # I think the paper uses step/epoch to refer to the number of episodes played

        if isinstance(smiles, str):
            score = self.scoring_function.get_final_score_for_step([smiles], self.num_episodes)
        elif isinstance(smiles, list):
            score = self.scoring_function.get_final_score_for_step(smiles, self.num_episodes)
        else:
            raise ValueError("Scoring error due to wrong dtype")

        return score.total_score[0]
