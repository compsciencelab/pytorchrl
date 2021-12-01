import gym
import numpy as np
import torch
import torch.nn as nn

from pytorchrl.agent.actors.utils import init
from pytorchrl.agent.actors.feature_extractors.mlp import MLP
from pytorchrl.agent.actors.feature_extractors.cnn import CNN


class DictNet(nn.Module):
    """
    Neural network architecture designed to deal with dict observation spaces.
    It extracts individual features independently with either a MLP or a CNN,
    and then combines the information with a final sequence of fully connected
    layers.

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    shared_hidden_sizes : list
        Hidden layers sizes used to combine multimodel features.
    activation : func
        Non-linear activation function.
    """
    def __init__(self, input_space, shared_hidden_sizes=[256, 256], output_size=256, activation=nn.ReLU):
        super(DictNet, self).__init__()

        assert isinstance(input_space, gym.spaces.Dict)

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init. constant_(
                x, 0), nn.init.calculate_gain('relu'))

        self.net_branches = nn.ModuleDict()
        for k in input_space.spaces:
            if len(input_space[k].shape) <= 2:
                net = MLP(input_space[k], activation=activation)
            elif len(input_space[k].shape) == 3:
                net = CNN(input_space[k], activation=activation)
            else:
                raise ValueError(
                    "Observation space with nested dicts not allowed.")
            self.net_branches[k] = net

        # Get concat output map size
        map_size = 0
        sample_obs = input_space.sample()
        for k in sample_obs:
            map_size += int(np.prod(
                self.net_branches[k](torch.from_numpy(
                    sample_obs[k]).float().unsqueeze(0)).shape))

        # Define shared layers
        layers = []
        sizes = [map_size] + shared_hidden_sizes + [output_size]
        for j in range(len(sizes) - 1):
            layers += [init_(nn.Linear(sizes[j], sizes[j + 1])), activation()]
        self.shared_net = nn.Sequential(*layers)

        self.train()

    def forward(self, inputs):
        """
        Forward pass Neural Network

        Parameters
        ----------
        inputs : torch.tensor
            Input data.

        Returns
        -------
        out : torch.tensor
            Output feature map.
        """

        assert isinstance(inputs, dict)

        output = []
        for k in inputs:
            x = inputs[k].view(inputs[k].size(0), -1)
            output.append(self.net_branches[k](x))

        output = torch.cat(output, dim=1)

        return self.shared_net(output)
