from typing import Tuple

import gym
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from pytorchrl.agent.actors.distributions import get_dist
from pytorchrl.agent.actors.utils import Scale, Unscale


# TODO: review prediction sizes when reward is and is not predicted!

class WorldModel(nn.Module):
    """
    Model-Based Actor class for Model-Based algorithms.

    It contains the dynamics network to predict the next state (and reward if selected). 

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space.
    hidden_size : int
        Hidden size number.
    standard_scaler : StandardScaler
        StandardScaler class instance.
    reward_function : func
        Reward function to be learned.
    """

    def __init__(self,
                 device,
                 input_space,
                 action_space,
                 standard_scaler,
                 hidden_size=64,
                 reward_function=None,
                 ) -> None:

        super(WorldModel, self).__init__()

        self.device = device
        self.input_space = input_space
        self.action_space = action_space
        self.reward_function = reward_function

        if reward_function is not None:
            self.predict = self.predict_given_reward
        else:
            self.predict = self.predict_learned_reward

        # Scaler for scaling training inputs
        self.standard_scaler = standard_scaler
        self.hidden_size = hidden_size
        self.model = self.create_dynamics()

    def create_dynamics(self):
        """
        Create a dynamics model and define it as class attribute under the name `name`.

        Parameters
        ----------
        name : str
            dynamics model name.
        """
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            input_layer = nn.Linear(self.input_space.shape[0] + self.action_space.n, out_features=self.hidden_size)
        else:
            input_layer = nn.Linear(self.input_space.shape[0] + self.action_space.shape[0], out_features=self.hidden_size)

        hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)

        if self.reward_function is None:
            num_outputs = self.input_space.shape[0] + 1
        else:
            num_outputs = self.input_space.shape[0]

        output_layer = get_dist("DeterministicMB")(num_inputs=self.hidden_size, num_outputs=num_outputs)
        self.activation = nn.ReLU
        dynamics_layers = [input_layer, self.activation(), hidden_layer, self.activation(), output_layer]

        if type(self.action_space) == gym.spaces.box.Box:
            self.scale = Scale(self.action_space)
            self.unscale = Unscale(self.action_space)
        else:
            self.scale = None
            self.unscale = None

        dynamics_net = nn.Sequential(*dynamics_layers)

        return dynamics_net

    @staticmethod
    def check_dynamics_weights(parameter1, parameter2):
        for p1, p2 in zip(parameter1, parameter2):
            if p1.data.ne(p2.data).sum() > 0:
                return False
        return True

    def reinitialize_dynamics_model(self):
        """
        Re-initializes the dynamics model, can be done before each new Model learning run.
        Might help in some environments to overcome over-fitting of the model!
        """
        old_weights = self.model.parameters()
        self.create_dynamics()
        self.model.to(self.device)
        new_weights = self.model.parameters()
        assert not self.check_dynamics_weights(old_weights, new_weights)

    def predict_learned_reward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Does the next state prediction and reward prediction with a learn reward function.

        Parameters
        ----------
        states : torch.Tensor
            Current state s
        actions : torch.Tensor
            Action taken in state s

        Returns
        -------
        next_states : torch.Tensor
            Next states.
        rewards : torch.Tensor
            Reward prediction.
        """

        if type(self.action_space) == gym.spaces.discrete.Discrete:
            actions = one_hot(actions, num_classes=self.action_space.n).squeeze(1)

        inputs = torch.cat((states, actions), dim=-1)

        # scale inputs based on recent batch scalings
        norm_inputs, _ = self.standard_scaler.transform(inputs)
        norm_predictions = self.model(norm_inputs)

        # inverse transform outputs
        predictions = self.standard_scaler.inverse_transform(norm_predictions)
        predictions[:, :-1] += states.to(self.device)

        next_states = predictions[:, :-1]
        rewards = predictions[:, -1].unsqueeze(-1)

        # TODO: add Termination function?

        return next_states, rewards

    def predict_given_reward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Does the next state prediction and calculates the reward given a reward function. 

        Parameters
        ----------
        states : torch.Tensor
            Current state s
        actions : torch.Tensor
            Action taken in state s

        Returns
        -------
        next_states : torch.Tensor
            Next states.
        rewards : torch.Tensor
            Calculated reward.
        """
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            actions = one_hot(actions, num_classes=self.action_space.n).squeeze(1)

        inputs = torch.cat((states, actions), dim=-1)

        # scale inputs based on recent batch scalings
        norm_inputs, _ = self.standard_scaler.transform(inputs)

        norm_predictions = self.model(norm_inputs)

        # inverse transform outputs
        predictions = self.standard_scaler.inverse_transform(norm_predictions)
        predictions += states.to(self.device)
        next_states = predictions

        rewards = self.reward_function(states, actions, next_states)

        # TODO: add Termination function?

        return next_states, rewards

