import gym
import copy
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
    """
    Batched custom Environment for Generative Chemistry RL.
    To be used when the scoring function is the bottleneck.
    """

    metadata = {"render.modes": ["human"]}

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
        self.max_scaffold_length = max([self.select_scaffold()[1] for _ in range(1000)])

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

        # Trackers
        self.current_decorations = ["" for _ in range(self.num_envs)]
        self.context = np.ones((self.num_envs, self.max_scaffold_length)) * self.vocabulary.encode_decoration_token("<pad>")
        self.context_length = np.zeros(self.num_envs)

    def step(self, action):
        """Execute one time step within the environment"""

        rew = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=np.bool)
        info = [{} for _ in range(self.num_envs)]

        finished = action == self.vocabulary.encode_decoration_token("$")
        done[finished] = True

        for i in range(self.num_envs):

            self.current_decorations[i] += self.vocabulary.decode_decoration_token(action[i])

            if finished[i]:

                # Join scaffold and decoration
                decorated_smile, molecule = self.join_scaffold_and_decorations(
                    self.vocabulary.decode_scaffold(self.context[i]),
                    self.vocabulary.remove_start_and_end_tokens(self.current_decorations[i]))

                # Compute score
                score = self.scoring_function(decorated_smile)

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

                rew[i] = reward

                # Update molecule
                env_info = {"molecule": decorated_smile or "invalid_smile"}
                env_info.update(score)
                info.append(env_info)

                # Reset finished env
                self.reset_single_env(i)

        observation = {
            "context": copy.copy(self.context),
            "context_length": copy.copy(self.context_length),
            "obs": action.reshape(self.num_envs, 1),
            "obs_length": np.ones(self.num_envs),
        }

        return observation, rew, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Return padded base molecule to match length `obs_length`.
        """

        # Define vectors
        obs = np.ones((self.num_envs, 1)) * self.vocabulary.encode_decoration_token("^")
        obs_length = np.ones(self.num_envs)

        # Fill up context and context_length
        for i in range(self.num_envs):
            scaffold, scaffold_length = self.select_scaffold()
            self.context[i, 0:scaffold_length] = scaffold
            self.context_length[i] = scaffold_length

        # Create observation
        observation = {
            "context": copy.copy(self.context),
            "context_length": copy.copy(self.context_length),
            "obs": obs,
            "obs_length": obs_length,
        }

        return observation

    def reset_single_env(self, num_env):
        """Reset environment in position num_env and return an the whole array of observations."""
        # Fill up context and context_length
        scaffold, scaffold_length = self.select_scaffold()
        self.context[num_env] = self.vocabulary.encode_decoration_token("<pad>")
        self.context[num_env, 0:scaffold_length] = scaffold
        self.context_length[num_env] = scaffold_length

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
        scaffold_length = len(scaffold)
        return scaffold, scaffold_length

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
