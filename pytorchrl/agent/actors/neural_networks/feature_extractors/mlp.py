import torch
import torch.nn as nn
from .utils import init


class MLP(nn.Module):
    """
    Multilayer Perceptron network

    Parameters
    ----------
    input_shape : tuple
        Shape input tensors.
    hidden_sizes : list
        Hidden layers sizes.
    activation : func
        Non-linear activation function.

    Attributes
    ----------
    feature_extractor : nn.Module
        Neural network feature extractor block.
    """
    def __init__(self, input_shape, hidden_sizes=[256, 256], activation=nn.ReLU):
        super(MLP, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init. constant_(x, 0), nn.init.calculate_gain('relu'))

        # Define feature extractor
        layers = []
        sizes = [input_shape[0]] + hidden_sizes
        for j in range(len(sizes) - 1):
            layers += [init_(nn.Linear(sizes[j], sizes[j + 1])), activation()]
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
        out = self.feature_extractor(inputs)
        return out
