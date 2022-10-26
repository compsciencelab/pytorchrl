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
    output_size : int
        Size of output feature map.
    activation : func
        Non-linear activation function.
    layer_norm: bool
        Use layer normalization.
    dropout: float
        Dropout probability.
    final_activation : bool
        Whether or not to apply activation function after last layer.
    """
    def __init__(self, input_space, hidden_sizes=[256, 256], output_size=256, activation=nn.ReLU, layer_norm=False, dropout=0.0, final_activation=True):
        super(MLP, self).__init__()

        if isinstance(input_space, gym.Space):
            input_shape = input_space.shape
        else:
            input_shape = input_space

        # Define feature extractor
        layers = []
        sizes = [np.prod(input_shape)] + hidden_sizes + [output_size]
        for j in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[j], sizes[j + 1])]
            if dropout > 0.0 and j < len(sizes) - 2:
                layers += [nn.Dropout(dropout)]
            if layer_norm and j < len(sizes) - 2:
                layers += [nn.LayerNorm(sizes[j + 1])]
            if j < len(sizes) - 2 or final_activation:
                layers += [activation()]
        self.feature_extractor = nn.Sequential(*layers)

        for layer in self.feature_extractor.children():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=get_gain(activation))
                layer.bias.data.zero_()

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
