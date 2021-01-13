import torch.nn as nn


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