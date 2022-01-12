import gym
import numpy as np
import torch
import torch.nn as nn
from pytorchrl.agent.actors.utils import init
from pytorchrl.agent.actors.feature_extractors.utils import get_gain


class CNN(nn.Module):
    """
    Convolutional Neural Network.

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    rgb_norm : bool
        Whether or not to divide input by 255.
    activation : func
        Non-linear activation function.
    final_activation : bool
        Whether or not to apply activation function after last layer.
    strides : list
        Convolutional layers strides.
    filters : list
        Convolutional layers number of filters.
    kernel_sizes : list
        Convolutional layers kernel sizes.
    output_sizes : list
        output hidden layers sizes.
    """
    def __init__(self,
                 input_space,
                 rgb_norm=True,
                 activation=nn.ReLU,
                 final_activation=True,
                 strides=[4, 2, 1],
                 filters=[32, 64, 64],
                 kernel_sizes=[8, 4, 3],
                 output_sizes=[256, 448]):

        super(CNN, self).__init__()

        self.rgb_norm = rgb_norm

        assert len(filters) == len(strides) and len(strides) == len(kernel_sizes)

        if isinstance(input_space, gym.Space):
            input_shape = input_space.shape
        else:
            input_shape = input_space

        if len(input_shape) != 3:
            raise ValueError("Trying to extract features with a CNN for an obs space with len(shape) != 3")

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), get_gain(activation))

        # Define CNN feature extractor
        layers = []
        filters = [input_shape[0]] + filters
        for j in range(len(filters) - 1):
            layers += [init_(nn.Conv2d(
                filters[j], filters[j + 1], stride=strides[j],
                kernel_size=kernel_sizes[j])), activation()]
        self.feature_extractor = nn.Sequential(*layers)

        # TODO. Final activation always ReLU
        activation = nn.ReLU
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), get_gain(activation))

        # Define final MLP layers
        feature_size = int(np.prod(self.feature_extractor(torch.randn(1, *input_shape)).shape))
        layers = []
        sizes = [feature_size] + output_sizes
        for jj in range(len(sizes) - 1):
            layers += [init_(nn.Linear(sizes[jj], sizes[jj + 1]))]
            if jj < len(sizes) - 2 or final_activation:
                layers += [activation()]
        self.head = nn.Sequential(*layers)

        for layer in self.feature_extractor.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=get_gain(activation))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=get_gain(activation))
                layer.bias.data.zero_()

        for layer in self.head.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=get_gain(activation))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
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

        if self.rgb_norm:
            inputs = inputs / 255.0

        out = self.feature_extractor(inputs)
        out = out.contiguous()
        out = out.view(inputs.size(0), -1)
        out = self.head(out)

        return out
