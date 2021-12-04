
import gym
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixupCNN(nn.Module):
    """
    A larger version of the IMPALA CNN with Fixup init.
    See Fixup: https://arxiv.org/abs/1901.09321.
    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    rgb_norm : bool
        Whether or not to divide input by 255.
    """
    def __init__(self, input_space, output_size=256, rgb_norm=True, activation=nn.ReLU, final_activation=True):

        super(FixupCNN, self).__init__()

        self.rgb_norm = rgb_norm

        if isinstance(input_space, gym.Space):
            input_shape = input_space.shape
        else:
            input_shape = input_space

        if len(input_shape) != 3:
            raise ValueError("Trying to extract features with Fixup CNN for "
                             "an obs space with len(shape) != 3")

        depth_in = input_shape[0]
        layers = []
        for depth_out in [16, 32, 32]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                FixupResidualModule(depth_out, 6),
                FixupResidualModule(depth_out, 6),
            ])
            depth_in = depth_out
        self.feature_extractor = nn.Sequential(*layers)

        # Define final layer
        feature_size = int(np.prod(self.feature_extractor(
            torch.randn(1, *input_space.shape)).shape))
        head = [nn.Linear(feature_size, output_size)]
        if final_activation:
            head += [nn.ReLU(inplace=True)]
        self.head = nn.Sequential(*head)

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

        out = self.feature_extractor(inputs).view(inputs.size(0), -1)
        return out


class FixupResidualModule(nn.Module):
    """
    FixupCNN resudial block.
    Parameters
    ----------
    depth : int
        Number of input feature maps.
    num_residual : int
        Number of residual feature blocks in the architecture
    """
    def __init__(self, depth, num_residual):
        super(FixupResidualModule, self).__init__()

        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)

        for p in self.conv1.parameters():
            p.data.mul_(1 / math.sqrt(num_residual))
        for p in self.conv2.parameters():
            p.data.zero_()

        self.bias1 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias2 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias3 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias4 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.scale = nn.Parameter(torch.ones([depth, 1, 1]))

    def forward(self, x):
        """
        Forward pass residual block
        Parameters
        ----------
        inputs : torch.tensor
            Input feature map.
        Returns
        -------
        out : torch.tensor
            Output feature map.
        """
        x = F.relu(x)
        out = x + self.bias1
        out = self.conv1(out)
        out = out + self.bias2
        out = F.relu(out)
        out = out + self.bias3
        out = self.conv2(out)
        out = out * self.scale
        out = out + self.bias4
        return out + x
