import gym
import numpy as np
from gym import spaces
from pytorchrl.envs.generative_chemistry.string_space import Char


class GenChemEnv(gym.Env):
    """Custom Environment for Generative Chemistry RL."""

    metadata = {'render.modes': ['human']}

    def __init__(self, scoring_function, scaffold, vocabulary, tokenizer, obs_length=50, **kwargs):
        super(GenChemEnv, self).__init__()

        self.num_episodes = 0
        self.tokenizer = tokenizer
        self.obs_length = obs_length
        self.vocabulary = vocabulary
        self.scoring_function = scoring_function

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(len(self.vocabulary))
        self.observation_space = Char(vocab=vocabulary.tokens(), max_length=1)

    def step(self, action):
        """Execute one time step within the environment"""

        if not isinstance(action, str):
            action = self.vocabulary.decode([action])[0]

        # TODO: action should be a single character

        # TODO: add character to current molecule
        self.current_molecule += action

        info = {}

        # TODO: if character is not $, return 0.0 reward
        if action != "$":
            reward = 0.0
            done = False

        # TODO: if character is $, evaluate molecule
        else:
            try:
                reward = self._scoring(self.tokenizer.untokenize(self.current_molecule))
                info.update({"molecules": self.tokenizer.untokenize(self.current_molecule)})
            except TypeError:
                reward = 0.0  # Invalid molecule
                info.update({"molecules": "invalid"})
            done = True

        new_obs = self.vocabulary.encode([action])

        return new_obs, reward, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.num_episodes += 1
        self.current_molecule = "^"
        return self.vocabulary.encode(self.current_molecule)

    def render(self, mode='human'):
        """Render the environment to the screen"""

        print(f'Current Molecule: {self.current_molecule}')
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
