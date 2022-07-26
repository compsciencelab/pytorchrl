import gym
import numpy as np
from gym import spaces
from collections import defaultdict, deque


class GenChemEnv(gym.Env):
    """Custom Environment for Generative Chemistry RL."""

    metadata = {'render.modes': ['human']}

    def __init__(self, scoring_function, vocabulary, tokenizer, max_length=200, **kwargs):
        super(GenChemEnv, self).__init__()

        self.num_episodes = 0
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocabulary = vocabulary
        self.scoring_function = scoring_function
        self.running_mean_valid_smiles = deque(maxlen=100)

        # Define action and observation space
        self.current_episode_length = 0
        self.action_space = gym.spaces.Discrete(len(self.vocabulary))
        self.observation_space = gym.spaces.Discrete(len(self.vocabulary))

    def step(self, action):
        """Execute one time step within the environment"""

        if not isinstance(action, str):
            action = self.vocabulary.decode([action])[0]

        info = {}
        self.current_episode_length += 1
        if self.current_episode_length == self.max_length:
            action = "$"
        self.current_molecule += action

        if action != "$":  # If character is not $, return 0.0 reward
            reward = 0.0
            done = False

        else:  # if action is $, evaluate molecule
            try:
                score = self._scoring(self.tokenizer.untokenize(self.current_molecule))
                reward = score.total_score[0]
                info.update({
                    "molecule": self.tokenizer.untokenize(self.current_molecule),
                    "regression_model": float(score.profile[0].score[0]),
                    "matching_substructure": float(score.profile[1].score[0]),
                    "custom_alerts": float(score.profile[2].score[0]),
                    "QED_score": float(score.profile[3].score[0]),
                    "raw_regression_model": float(score.profile[4].score[0]),
                })
                self.running_mean_valid_smiles.append(1.0)
            except TypeError:
                reward = 0.0  # Invalid molecule
                info.update({
                    "molecule": "invalid",
                    "regression_model": 0.0,
                    "matching_substructure": 0.0,
                    "custom_alerts": 0.0,
                    "QED_score": 0.0,
                    "raw_regression_model": 0.0,
                })
                self.running_mean_valid_smiles.append(0.0)
            done = True

        # Update valid smiles tracker
        info.update({
            "valid_smiles": float((sum(self.running_mean_valid_smiles) / len(self.running_mean_valid_smiles))
                                  if len(self.running_mean_valid_smiles) != 0.0 else 0.0)})

        new_obs = self.vocabulary.encode([action])

        return new_obs, reward, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.num_episodes += 1
        self.current_molecule = "^"
        self.current_episode_length = 0
        return self.vocabulary.encode(self.current_molecule)

    def render(self, mode='human'):
        """Render the environment to the screen"""

        print(f'Current Molecule: {self.current_molecule}')
        print(f'Vocabulary: {self.vocabulary._tokens}')

    def _scoring(self, smiles):
        """Return scoring metric."""

        # I think the paper uses step/epoch to refer to the number of episodes played

        if isinstance(smiles, str):
            # score = self.scoring_function.get_final_score_for_step([smiles], self.num_episodes)
            score = self.scoring_function.get_final_score([smiles])
        elif isinstance(smiles, list):
            # score = self.scoring_function.get_final_score_for_step(smiles, self.num_episodes)
            score = self.scoring_function.get_final_score(smiles)
        else:
            raise ValueError("Scoring error due to wrong dtype")

        return score
