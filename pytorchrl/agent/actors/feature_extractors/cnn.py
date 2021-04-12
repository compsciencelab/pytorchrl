import torch
import torch.nn as nn
from pytorchrl.agent.actors.utils import init


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
    strides : list
        Convolutional layers strides.
    filters : list
        Convolutional layers number of filters.
    kernel_sizes : list
        Convolutional layers kernel sizes.
    """
    def __init__(self,
                 input_space,
                 rgb_norm=True,
                 activation=nn.ReLU,
                 strides=[4, 2, 1],
                 filters=[32, 64, 32],
                 kernel_sizes=[8, 4, 3]):

        super(CNN, self).__init__()

        input_shape = input_space.shape

        if len(input_shape) != 3:
            raise ValueError("Trying to extract features with a CNN for an obs space with len(shape) != 3")

        assert len(filters) == len(strides) and len(strides) == len(kernel_sizes)

        try:
            gain = nn.init.calculate_gain(activation.__name__.lower())
        except Exception:
            gain = 1.0

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        # Define feature extractor
        layers = []
        filters = [input_shape[0]] + filters
        for j in range(len(filters) - 1):
            layers += [init_(nn.Conv2d(
                filters[j], filters[j + 1], stride=strides[j],
                kernel_size=kernel_sizes[j])), activation()]
        self.feature_extractor = nn.Sequential(*layers)
        self.train()

        self.rgb_norm = rgb_norm

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
