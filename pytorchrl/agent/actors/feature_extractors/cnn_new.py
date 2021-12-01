from abc import ABC
import numpy as np
import gym
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorchrl.agent.actors.utils import init



def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1


class CNN(nn.Module, ABC):

    def __init__(self, input_space):
        super(CNN, self).__init__()

        if isinstance(input_space, gym.Space):
            input_shape = input_space.shape
        else:
            input_shape = input_space

        c, w, h = input_shape
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc1 = nn.Linear(in_features=flatten_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=448)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        self.fc1.bias.data.zero_()
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        self.fc2.bias.data.zero_()

        # nn.init.orthogonal_(self.extra_policy_fc.weight, gain=np.sqrt(0.1))
        # self.extra_policy_fc.bias.data.zero_()
        # nn.init.orthogonal_(self.extra_value_fc.weight, gain=np.sqrt(0.1))
        # self.extra_value_fc.bias.data.zero_()

        # nn.init.orthogonal_(self.policy.weight, gain=np.sqrt(0.01))
        # self.policy.bias.data.zero_()
        # nn.init.orthogonal_(self.int_value.weight, gain=np.sqrt(0.01))
        # self.int_value.bias.data.zero_()
        # nn.init.orthogonal_(self.ext_value.weight, gain=np.sqrt(0.01))
        # self.ext_value.bias.data.zero_()

    def forward(self, inputs):
        x = inputs / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = F.relu(self.fc2(x))
        return out
