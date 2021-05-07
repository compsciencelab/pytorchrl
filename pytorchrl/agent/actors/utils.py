import torch
import torch.nn as nn


class Scale(nn.Module):
    """
    Maps inputs from [space.low, space.high] range to [-1, 1] range.

    Parameters
    ----------
    space : gym.Space
        Space to map from.

    Attributes
    ----------
    low : torch.tensor
        Lower bound for unscaled Space.
    high : torch.tensor
        Upper bound for unscaled Space.
    """
    def __init__(self, space):
        super(Scale, self).__init__()
        self.register_buffer("low", torch.from_numpy(space.low))
        self.register_buffer("high", torch.from_numpy(space.high))

    def forward(self, x):
        """
        Maps x from [space.low, space.high] to [-1, 1].

        Parameters
        ----------
        x : torch.tensor
            Input to be scaled
        """
        return 2.0 * ((x - self.low) / (self.high - self.low)) - 1.0


class Unscale(nn.Module):
    """
    Maps inputs from [-1, 1] range to [space.low, space.high] range.

    Parameters
    ----------
    space : gym.Space
        Space to map from.

    Attributes
    ----------
    low : torch.tensor
        Lower bound for unscaled Space.
    high : torch.tensor
        Upper bound for unscaled Space.
    """
    def __init__(self, space):
        super(Unscale, self).__init__()
        self.register_buffer("low", torch.from_numpy(space.low))
        self.register_buffer("high", torch.from_numpy(space.high))

    def forward(self, x):
        """
        Maps x from [-1, 1] to [space.low, space.high].

        Parameters
        ----------
        x : torch.tensor
            Input to be unscaled
        """
        return self.low + (0.5 * (x + 1.0) * (self.high - self.low))


def init(module, weight_init, bias_init, gain=1):
    """
    Parameters
    ----------
    module : nn.Module
        nn.Module to initialize.
    weight_init : func
        Function to initialize module weights.
    bias_init : func
        Function to initialize module biases.
    Returns
    -------
    module : nn.Module
        Initialized module
    """
    weight_init(module.weight.data, gain=gain)
    weight_init(module.weight.data)
    bias_init(module.bias.data)
    return module
