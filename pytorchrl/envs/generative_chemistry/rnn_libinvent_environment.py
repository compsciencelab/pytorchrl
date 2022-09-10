import gym
import numpy as np
from gym import spaces
from collections import defaultdict, deque

### TODO: make eveything work without loading the prior

# TODO: define obs as space.dict

# TODO: Define an optional "scaffold_list" parameter, which allows to provide scaffolds. If scaffolds available,
# randomly pick one at the beginning of every episode.

# TODO: code reset function

# TODO: code step function

# TODO: during the episode, provide {"scaffold": scaffold, "decoration": last token} as obs,
#  and expect a decoration next token action.

# TODO: For data collection is straightforward. In the memory net, compute the scaffold hidden state only if Done!

# TODO: for gradient computation, need to delve into the code a bit more.

### TODO: make eveything work without with the prior


### TODO: allow training our own prior


class GenChemEnv(gym.Env):
    """Custom Environment for Generative Chemistry RL."""

    metadata = {'render.modes': ['human']}

    def __init__(self, scoring_function, vocabulary, scaffolds,  max_length=200):
        super(GenChemEnv, self).__init__()

        self.num_episodes = 0
        self.scaffolds = scaffolds
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.current_episode_length = 0
        self.concatenate_obs = concatenate_obs
        self.scoring_function = scoring_function
        self.running_mean_valid_smiles = deque(maxlen=100)

        # Break down scaffolds into tokens
        import ipdb; ipdb.set_trace()  # TODO: convert tokenized_scaffolds to arrays
        self.tokenized_scaffolds = [vocabulary.encode(tokenizer.tokenize(i)) for i in self.scaffolds]
        self.max_scaffold_length = max([len(i) for i in self.tokenized_scaffolds])

        # Define action and observation space
        import ipdb; ipdb.set_trace()
        scaffold_space = gym.spaces.Discrete(len(self.vocabulary))
        scaffold_space.shape = (self.max_scaffold_length,)  # Ugly hack
        decoration_space = gym.spaces.Discrete(len(self.vocabulary))
        self.observation_space = gym.spaces.Dict({
            "scaffold": scaffold_space,
            "decoration": decoration_space,
        })
        self.action_space = gym.spaces.Discrete(len(self.vocabulary))

        if concatenate_obs:
            pass

    def step(self, action):
        """Execute one time step within the environment"""

        import ipdb; ipdb.set_trace()

        if not isinstance(action, str):
            action = self.vocabulary.decode([action])[0]

        info = {}
        self.current_episode_length += 1
        if self.current_episode_length == self.max_length - 1:
            action = "$"
        self.current_molecule_str += action

        if action != "$":  # If character is not $, return 0.0 reward
            reward = 0.0
            done = False

        else:  # if action is $, evaluate molecule

            score = self.scoring_function(self.tokenizer.untokenize(self.current_molecule_str))

            assert isinstance(score, dict), "scoring_function has to return a dict"

            assert "score" in score.keys() or "reward" in score.keys(), \
                "scoring_function outputs requires at lest the keyword ´score´ or ´reward´"

            # Get reward
            if "reward" in score.keys():
                reward = score["reward"]
            else:
                reward = score["score"]

            # If score contain field "Valid", update counter
            if "valid_smile" in score.keys():
                valid = score["valid_smile"]
                if valid:
                    self.running_mean_valid_smiles.append(1.0)
                else:
                    self.running_mean_valid_smiles.append(0.0)

            # Update molecule
            info.update({"molecule": self.tokenizer.untokenize(self.current_molecule_str)})

            # Update valid smiles tracker
            info.update({"valid_smile": float((sum(self.running_mean_valid_smiles) / len(
                self.running_mean_valid_smiles)) if len(self.running_mean_valid_smiles) != 0.0 else 0.0)})

            # Update info with remaining values
            info.update(score)
            done = True

        new_obs = self.vocabulary.encode([action])
        if self.concatenate_obs:
            self.current_molecule_np[self.current_episode_length] = new_obs
            return self.current_molecule_np, reward, done, info

        return new_obs, reward, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.num_episodes += 1
        self.current_molecule = "^"
        self.current_episode_length = 0
        obs = {
            "scaffold": self.tokenized_scaffolds[0],
            "decoration":  self.vocabulary.encode(self.current_molecule),
        }
        return obs

    def render(self, mode='human'):
        """Render the environment to the screen"""

        print(f'Current Molecule: {self.current_molecule}')
        print(f'Vocabulary: {self.vocabulary._tokens}')
