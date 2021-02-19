import torch
import torch.nn as nn
from .utils import init


class CNN(nn.Module):
    """
    Convolutional Neural Network.

    Parameters
    ----------
    input_shape : tuple
        Shape input tensors.
    activation : func
        Non-linear activation function.
    strides : list
        Convolutional layers strides.
    filters : list
        Convolutional layers number of filters.
    kernel_sizes : list
        Convolutional layers kernel sizes.
    """
    def __init__(self,
                 input_shape,
                 activation=nn.ReLU,
                 strides=[4, 2, 1],
                 filters=[32, 64, 32],
                 kernel_sizes=[8, 4, 3]):
        super(CNN, self).__init__()

        assert len(filters) == len(strides) and len(strides) == len(kernel_sizes)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

        # Define feature extractor
        layers = []
        filters = [input_shape[0]] + filters
        for j in range(len(filters) - 1):
            layers += [init_(nn.Conv2d(
                filters[j], filters[j + 1], stride=strides[j],
                kernel_size=kernel_sizes[j])), activation()]
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
        out = self.feature_extractor(inputs / 255.0)
        return out
