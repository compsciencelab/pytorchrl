import gym
import numpy as np
from gym import spaces

from pytorchrl.envs.generative_chemistry.string_space import Char


# TODO: review if smiles have to start and end with special characters!


class GenChemEnv(gym.Env):
    """Custom Environment for Generative Chemistry RL."""

    metadata = {'render.modes': ['human']}

    def __init__(self, scoring_function, base_molecule, vocabulary, obs_length=50, **kwargs):
        super(GenChemEnv, self).__init__()

        self.num_episodes = 0
        self.obs_length = obs_length
        self.vocabulary = vocabulary
        self.base_molecule = base_molecule
        self.scoring_function = scoring_function

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = Char(vocab=vocabulary.tokens(), max_length=obs_length)
        self.observation_space = Char(vocab=vocabulary.tokens(), max_length=obs_length)

        # Make sure base molecule is a string
        # TODO: add more sanity checks to base molecule, to make sure is valid
        assert isinstance(base_molecule, str), "Base molecule is not a sting"

    def step(self, action):
        """Execute one time step within the environment"""

        self.new_molecule = action
        reward = self._scoring(self.new_molecule)
        info = {}
        done = True
        new_obs = None  # Does not matter

        return new_obs, reward, info, done

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.num_episodes += 1
        return self.base_molecule.ljust(self.obs_length, self.vocabulary._tokens[0])

    def render(self, mode='human', close=False):
        """Render the environment to the screen"""

        print(f'Scaffold: {self.base_molecule}')
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

        return score
