import gym
import numpy as np
import torch.nn as nn
from pytorchrl.agent.actors.utils import init
from pytorchrl.agent.actors.feature_extractors.utils import get_gain


class MLP(nn.Module):
    """
    Multilayer Perceptron network

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    hidden_sizes : list
        Hidden layers sizes.
    activation : func
        Non-linear activation function.
    final_activation : bool
        Whether or not to apply activation function after last layer.
    """
    def __init__(self, input_space, hidden_sizes=[256, 256], output_size=256, activation=nn.ReLU, final_activation=True):
        super(MLP, self).__init__()

        if isinstance(input_space, gym.Space):
            input_shape = input_space.shape
        else:
            input_shape = input_space

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init. constant_(x, 0), get_gain(activation))

        # Define feature extractor
        layers = []
        sizes = [np.prod(input_shape)] + hidden_sizes + [output_size]
        for j in range(len(sizes) - 1):
            layers += [init_(nn.Linear(sizes[j], sizes[j + 1]))]
            if not (j == len(sizes) - 2 and final_activation):
                layers += [activation()]
        self.feature_extractor = nn.Sequential(*layers)

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
        inputs = inputs.view(inputs.size(0), -1)
        out = self.feature_extractor(inputs)
        return out
