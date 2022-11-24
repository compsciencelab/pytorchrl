import gym
import random
import numpy as np
import rdkit as Chem
from gym import spaces
from collections import defaultdict, deque
from reinvent_chemistry import Conversions
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints
from reinvent_chemistry.library_design.reaction_filters.reaction_filter import ReactionFilter, \
    ReactionFilterConfiguration
from pytorchrl.envs.generative_chemistry.diversity_filter.no_filter_with_penalty import NoFilterWithPenalty


class GenChemEnv(gym.Env):
    """Custom Environment for Generative Chemistry RL."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, scoring_function, vocabulary, scaffolds, randomize_scaffolds=False, max_length=200,
                 reactions=[]):
        super(GenChemEnv, self).__init__()

        self.num_episodes = 0
        self.reactions = reactions
        self.scaffolds = scaffolds
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.current_episode_length = 0
        self.scoring_function = scoring_function
        self.randomize_scaffolds = randomize_scaffolds
        self.running_mean_valid_smiles = deque(maxlen=1)

        self._bond_maker = BondMaker()
        self._conversion = Conversions()
        self._attachment_points = AttachmentPoints()

        # Check maximum possible scaffold length
        self.max_scaffold_length = max([len(self.select_scaffold()) for _ in range(1000)])

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(len(self.vocabulary.decoration_vocabulary))
        scaffold_space = gym.spaces.Discrete(len(self.vocabulary.scaffold_vocabulary))
        scaffold_length = gym.spaces.Discrete(self.max_scaffold_length)
        decoration_space = gym.spaces.Discrete(len(self.vocabulary.decoration_vocabulary))
        decoration_length = gym.spaces.Discrete(self.max_length)
        full_decoration_space = gym.spaces.Discrete(len(self.vocabulary.decoration_vocabulary))
        full_decoration_length = gym.spaces.Discrete(self.max_length)

        # Ugly hack
        scaffold_space._shape = (self.max_scaffold_length,)
        # full_decoration_space._shape = (self.max_length,)
        decoration_space._shape = (1,)

        self.observation_space = gym.spaces.Dict({
            "context": scaffold_space,
            "context_length": scaffold_length,
            "obs": decoration_space,
            "obs_length": decoration_length,
            # "full_obs": full_decoration_space,
            # "full_obs_length": full_decoration_length,
        })

        # Reaction Filters
        reaction_filter_conf = {"type": "selective", "reactions": reactions}
        reaction_filter_conf = ReactionFilterConfiguration(
            type=reaction_filter_conf["type"],
            reactions=reaction_filter_conf["reactions"],
            reaction_definition_file=None)
        self.reaction_filter = ReactionFilter(reaction_filter_conf)

        # Diversity Filter: penalizes the score by 0.5 if a previously seen compound is proposed.
        self.diversity_filter = NoFilterWithPenalty()

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

            decorated_smile, molecule = self.join_scaffold_and_decorations(
                self.vocabulary.decode_scaffold(self.padded_scaffold),
                self.vocabulary.remove_start_and_end_tokens(self.current_decoration)
            )

            # Compute score
            score = self.scoring_function(decorated_smile)

            # Sanity check
            assert isinstance(score, dict), "scoring_function has to return a dict"
            assert "score" in score.keys() or "reward" in score.keys(), \
                "scoring_function outputs requires at lest the keyword ´score´ or ´reward´"

            # Apply reaction filters
            score.update({"reaction_scores": 0.0})
            if molecule:
                self.apply_reaction_filters(molecule, score)

            # Get reward
            reward = score["reward"] if "reward" in score.keys() else score["score"]

            # Adjust reward with diversity filter
            reward = self.diversity_filter.update_score(reward, decorated_smile)

            # If score contain field "Valid", update counter
            if "valid_smile" in score.keys():
                valid = score["valid_smile"]
                self.running_mean_valid_smiles.append(1.0) if valid else \
                    self.running_mean_valid_smiles.append(0.0)

            # Update molecule
            info.update({"molecule": decorated_smile or "invalid_smile"})

            # Update info with remaining values
            info.update(score)

            # Update valid smiles tracker
            info.update({"valid_smile": float((sum(self.running_mean_valid_smiles) / len(
                self.running_mean_valid_smiles)) * 100 if len(self.running_mean_valid_smiles) != 0.0 else 0.0)})

            done = True

        self.padded_current_decoration[self.current_decoration_length - 1] = \
            self.vocabulary.encode_decoration_token(action)

        next_obs = {
            "context": self.padded_scaffold,
            "context_length": np.array(self.scaffold_length),
            "obs": np.array(self.vocabulary.encode_decoration_token(action)).reshape(1),
            "obs_length": np.array(1),
            # "full_obs":  self.padded_current_decoration,
            # "full_obs_length": np.array(self.current_decoration_length),
        }

        return next_obs, reward, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.num_episodes += 1

        self.scaffold_length = np.Inf
        while self.scaffold_length > self.max_scaffold_length:
            self.scaffold = self.select_scaffold()
            self.scaffold_length = len(self.scaffold)

        self.padded_scaffold = self.vocabulary.encode_scaffold_token("<pad>") * np.ones(self.max_scaffold_length)
        self.padded_scaffold[0:self.scaffold_length] = self.scaffold

        self.current_decoration = "^"
        self.padded_current_decoration = self.vocabulary.encode_decoration_token("<pad>") * np.ones(self.max_length)
        self.padded_current_decoration[0] = self.vocabulary.encode_decoration_token(self.current_decoration)
        self.current_decoration_length = 1

        # "decoration":  self.padded_current_decoration,
        # "decoration_length": np.array(self.current_decoration_length),

        obs = {
            "context": self.padded_scaffold,
            "context_length": np.array(self.scaffold_length),
            "obs": np.array(self.vocabulary.encode_decoration_token(self.current_decoration)).reshape(1),
            "obs_length": np.array(1),
            # "full_obs":  self.padded_current_decoration,
            # "full_obs_length": np.array(self.current_decoration_length),
        }

        return obs

    def render(self, mode="human"):
        """Render the environment to the screen"""

        print(f"Current Molecule: {self.current_molecule}")
        print(f"Vocabulary: {self.vocabulary._tokens}")

    def select_scaffold(self):
        scaffold = random.choice(self.scaffolds)
        if self.randomize_scaffolds and len(self.reactions) == 0:
            mol = self._conversion.smile_to_mol(scaffold)
            scaffold = self._bond_maker.randomize_scaffold(mol)  # randomize
        scaffold = self._attachment_points.remove_attachment_point_numbers(scaffold)
        scaffold = self.vocabulary.encode_scaffold(scaffold)
        return scaffold

    def join_scaffold_and_decorations(self, scaffold, decorations):
        scaffold = self._attachment_points.add_attachment_point_numbers(scaffold, canonicalize=False)
        molecule = self._bond_maker.join_scaffolds_and_decorations(scaffold, decorations)
        smile = self._conversion.mol_to_smiles(molecule) if molecule else None
        return smile, molecule

    def apply_reaction_filters(self, mol, final_score):
        reaction_scores = [self.reaction_filter.evaluate(mol) if mol else 0.0]
        reward = final_score["reward"] if "reward" in final_score.keys() else final_score["score"]
        final_score["reward"] = float(reward * np.array(reaction_scores))
        final_score["reaction_scores"] = float(np.array(reaction_scores))
        return final_score
