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
from pytorchrl.agent.env import BatchedEnv


class BatchedGenChemEnv(BatchedEnv):
    """Custom Environment for Generative Chemistry RL."""

    metadata = {'render.modes': ['human']}

    def __init__(self, scoring_function, vocabulary, scaffolds, randomize_scaffolds=False, max_length=200,
                 reactions=[], num_envs=1):
        super(BatchedGenChemEnv, self).__init__(num_envs=num_envs)

        self.num_envs = num_envs
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

        # Define action and observation space of a single environment
        self.action_space = gym.spaces.Discrete(len(self.vocabulary.decoration_vocabulary))
        scaffold_space = gym.spaces.Discrete(len(self.vocabulary.scaffold_vocabulary))
        scaffold_length = gym.spaces.Discrete(self.max_scaffold_length)
        decoration_space = gym.spaces.Discrete(len(self.vocabulary.decoration_vocabulary))
        decoration_length = gym.spaces.Discrete(self.max_length)

        # Ugly hack
        scaffold_space._shape = (self.num_envs, self.max_scaffold_length)
        decoration_space._shape = (self.num_envs, 1)

        self.observation_space = gym.spaces.Dict({
            "context": scaffold_space,
            "context_length": scaffold_length,
            "obs": decoration_space,
            "obs_length": decoration_length,
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

        observation = {
            "context": np.zeros((self.num_envs, self.max_scaffold_length)),
            "context_length": np.zeros(self.num_envs) + 10,
            "obs": np.zeros((self.num_envs, 1)) + 2,
            "obs_length": np.zeros(self.num_envs) + 1,
        }
        rew = np.zeros((self.num_envs, ), dtype=np.float32)
        done = np.zeros((self.num_envs, ), dtype=np.bool)
        info = [{} for _ in range(self.num_envs)]

        return observation, rew, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Return padded base molecule to match length `obs_length`.
        """

        # Initial observation
        observation = {
            "context": np.zeros((self.num_envs, self.max_scaffold_length)),
            "context_length": np.zeros(self.num_envs) + 10,
            "obs": np.zeros((self.num_envs, 1)) + 2,
            "obs_length": np.zeros(self.num_envs) + 1,
        }

        return observation

    def render(self, mode='human'):
        """Render the environment to the screen"""

        print(f'Current Molecule: {self.current_molecule}')
        print(f'Vocabulary: {self.vocabulary._tokens}')

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
