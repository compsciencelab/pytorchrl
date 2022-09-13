import gym
import random
import numpy as np
from gym import spaces
from collections import defaultdict, deque


from reinvent_chemistry import Conversions
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints
from running_modes.reinforcement_learning.dto.sampled_sequences_dto import SampledSequencesDTO


### TODO: make eveything work without loading the prior

# TODO: define obs as space.dict

# TODO: scaffolds and decorations are of fixed size with padding

# TODO: In both cases create a np vector with padding tokens and fill it up with data

# TODO: Then provide the real length of the scaffold/decoration in the obs

### TODO: make eveything work without with the prior


### TODO: allow training our own prior


class GenChemEnv(gym.Env):
    """Custom Environment for Generative Chemistry RL."""

    metadata = {'render.modes': ['human']}

    def __init__(self, scoring_function, vocabulary, scaffolds, max_length=200):
        super(GenChemEnv, self).__init__()

        self.num_episodes = 0
        self.scaffolds = scaffolds
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.current_episode_length = 0
        self.scoring_function = scoring_function
        self.running_mean_valid_smiles = deque(maxlen=100)

        # Break down scaffolds into tokens
        self.clean_scaffolds = [self._attachment_points.remove_attachment_point_numbers(scaffold) for scaffold in self.scaffolds]
        self.vectorised_scaffolds = [vocabulary.encode_scaffold(i) for i in self.clean_scaffolds]
        self.max_scaffold_length = max([vocabulary.count_scaffold_tokens(i) for i in self.clean_scaffolds])

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(len(self.vocabulary.decoration_vocabulary))
        scaffold_space = gym.spaces.Discrete(len(self.vocabulary.scaffold_vocabulary))
        scaffold_length = gym.spaces.Discrete(self.max_scaffold_length)
        decoration_space = gym.spaces.Discrete(len(self.vocabulary.decoration_vocabulary))
        decoration_length = gym.spaces.Discrete(self.max_length)

        # Ugly hack
        scaffold_space._shape = (self.max_scaffold_length, )
        decoration_space._shape = (self.max_length, )

        self.observation_space = gym.spaces.Dict({
            "scaffold": scaffold_space,
            "scaffold_length": scaffold_length,
            "decoration": decoration_space,
            "decoration_length": decoration_length,
        })

        self._bond_maker = BondMaker()
        self._conversion = Conversions()
        self._attachment_points = AttachmentPoints()

    def step(self, action):
        """Execute one time step within the environment"""

        info = {}
        self.current_decoration_length += 1
        action = "$" if self.current_decoration_length == self.max_length - 1 else \
            self.vocabulary.decode_decoration_token(action)
        self.current_decoration += action

        if action != "$":  # If character is not $, return 0.0 reward
            reward = 0.0
            done = False

        else:  # if action is $, evaluate molecule

            import ipdb; ipdb.set_trace()
            decorated_smile = self.join_scaffold_and_decorations(
                self.vocabulary.remove_start_and_end_tokens(self.padded_scaffold),
                self.vocabulary.remove_start_and_end_tokens(self.current_decoration)
            )

            # Compute score
            score = self.scoring_function(decorated_smile)

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
            info.update({"molecule": self.vocabulary.remove_start_and_end_tokens(self.current_decoration)})

            # Update valid smiles tracker
            info.update({"valid_smile": float((sum(self.running_mean_valid_smiles) / len(
                self.running_mean_valid_smiles)) if len(self.running_mean_valid_smiles) != 0.0 else 0.0)})

            # Update info with remaining values
            info.update(score)
            done = True

        self.padded_current_decoration[self.current_decoration_length - 1] = \
            self.vocabulary.encode_decoration_token(action)

        next_obs = {
            "scaffold": self.padded_scaffold,
            "scaffold_length": np.array(self.scaffold_length),

            # "decoration":  self.padded_current_decoration,
            # "decoration_length": np.array(self.current_decoration_length),

            "decoration": np.array(self.vocabulary.encode_decoration_token(action)).reshape(1),
            "decoration_length": np.array(1),
        }

        return next_obs, reward, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.num_episodes += 1

        self.scaffold = random.choice(self.vectorised_scaffolds)
        self.scaffold_length = len(self.scaffold)
        self.padded_scaffold = self.vocabulary.encode_scaffold_token('<pad>') * np.ones(self.max_scaffold_length)
        self.padded_scaffold[0:self.scaffold_length] = self.scaffold

        self.current_decoration = "^"
        self.padded_current_decoration = self.vocabulary.encode_decoration_token('<pad>') * np.ones(self.max_length)
        self.padded_current_decoration[0] = self.vocabulary.encode_decoration_token(self.current_decoration)
        self.current_decoration_length = 1

        obs = {
            "scaffold": self.padded_scaffold,
            "scaffold_length": np.array(self.scaffold_length),

            # "decoration":  self.padded_current_decoration,
            # "decoration_length": np.array(self.current_decoration_length),

            "decoration": np.array(self.vocabulary.encode_decoration_token(self.current_decoration)).reshape(1),
            "decoration_length": np.array(1),

        }

        return obs

    def render(self, mode='human'):
        """Render the environment to the screen"""

        print(f'Current Molecule: {self.current_molecule}')
        print(f'Vocabulary: {self.vocabulary._tokens}')

    def join_scaffold_and_decorations(self, scaffold, decorations):
        scaffold = self._attachment_points.add_attachment_point_numbers(scaffold, canonicalize=False)
        molecule = self._bond_maker.join_scaffolds_and_decorations(scaffold, decorations)
        smile = self._conversion.mol_to_smiles(molecule)
        return smile
