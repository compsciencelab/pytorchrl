import gym
import numpy as np
from gym import spaces
from collections import defaultdict, deque


class GenChemEnv(gym.Env):
    """Custom Environment for Generative Chemistry RL."""

    metadata = {'render.modes': ['human']}

    def __init__(self, scoring_function, vocabulary, max_length=200):
        super(GenChemEnv, self).__init__()

        self.num_episodes = 0
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.scoring_function = scoring_function
        self.running_mean_valid_smiles = deque(maxlen=100)

        # Define action and observation space
        self.current_episode_length = 0
        self.action_space = gym.spaces.Discrete(len(self.vocabulary))
        self.observation_space = gym.spaces.Discrete(len(self.vocabulary))
        self.action_space.shape = (max_length,)
        self.observation_space.shape = (max_length,)
        self.current_molecule_np = -1 * np.ones(max_length)

    def step(self, action):
        """Execute one time step within the environment"""

        info = {}
        self.current_episode_length += 1
        action = "$" if self.current_episode_length == self.max_length - 1 else \
            self.vocabulary.decode_token(action)
        self.current_molecule_str += action

        if action != "$":  # If character is not $, return 0.0 reward
            reward = 0.0
            done = False

        else:  # if action is $, evaluate molecule

            # Compute score
            score = self.scoring_function(self.vocabulary.remove_start_and_end_tokens(self.current_molecule_str))

            # Sanity check
            assert isinstance(score, dict), "scoring_function has to return a dict"
            assert "score" in score.keys() or "reward" in score.keys(), \
                "scoring_function outputs requires at lest the keyword ´score´ or ´reward´"

            # Get reward
            reward = score["reward"] if "reward" in score.keys() else score["score"]

            # If score contain field "Valid", update counter
            if "valid_smile" in score.keys():
                valid = score["valid_smile"]
                self.running_mean_valid_smiles.append(1.0) if valid else \
                    self.running_mean_valid_smiles.append(0.0)

            # Update molecule
            info.update({"molecule": self.vocabulary.remove_start_and_end_tokens(self.current_molecule_str)})

            # Update valid smiles tracker
            info.update({"valid_smile": float((sum(self.running_mean_valid_smiles) / len(
                self.running_mean_valid_smiles)) * 100 if len(self.running_mean_valid_smiles) != 0.0 else 0.0)})

            # Update info with remaining values
            info.update(score)
            done = True

        new_obs = self.vocabulary.encode_token(action)
        self.current_molecule_np[self.current_episode_length] = new_obs
        return self.current_molecule_np, reward, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.num_episodes += 1
        self.current_molecule_str = "^"
        self.current_episode_length = 0
        self.current_molecule_np = -1 * np.ones(self.max_length)
        self.current_molecule_np[self.current_episode_length] = self.vocabulary.encode_token("^")
        return self.current_molecule_np

    def render(self, mode='human'):
        """Render the environment to the screen"""

        print(f'Current Molecule: {self.current_molecule}')
        print(f'Vocabulary: {self.vocabulary._tokens}')
