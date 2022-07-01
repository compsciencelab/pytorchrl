import torch.nn as nn


def get_gain(activation):
    name = activation.__name__.lower()
    if name == "leakyrelu":
        name = "leaky_relu"
    gain = nn.init.calculate_gain(name)
    return gain
