import numpy as np
import torch
import torch.nn as nn
from pytorchrl.agent.actors.utils import init


class GruNet(nn.Module):
    """
    Base Neural Network class.

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    feature_extractor : nn.Module
        PyTorch nn.Module used as the features extraction block.
    feature_extractor_kwargs : dict
        Keyword arguments for the feature extractor network.
    recurrent : bool
        Whether to use recurrency or not.
    """
    def __init__(self, input_size, activation=nn.ReLU):

        super(GruNet, self).__init__()

        try:
            gain = nn.init.calculate_gain(activation.__name__.lower())
        except Exception:
            gain = 1.0

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self._num_outputs = input_size

        # Apply recurrency to extracted features
        self.gru = nn.GRU(input_size, input_size)
        self.final_layer = init_(nn.Linear(self._num_outputs, self._num_outputs))
        self.final_activation = activation()

        self.train()

    @property
    def num_outputs(self):
        """Output feature map size (as in np.prod(self.output_shape))."""
        return self._num_outputs

    @property
    def recurrent_hidden_state_size(self):
        """Recurrent hidden state size"""
        return self._num_outputs

    def _forward_gru(self, x, hxs, done):
        """
        Fast forward pass GRU network.
        from Ilya Kostrikov.PyTorch Implementations of Reinforcement Learning Algorithms.
        https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail. 2018
        Parameters
        ----------
        x : torch.tensor
            Feature map obtained from environment observation.
        hxs : torch.tensor
            Current recurrent hidden state.
        done : torch.tensor
            Current done tensor, indicating if episode has finished.
        Returns
        -------
        x : torch.tensor
            Feature map obtained after GRU.
        hxs : torch.tensor
            Updated recurrent hidden state.
        """

        masks = 1 - done

        if x.size(0) == hxs.size(0):
            self.gru.flatten_parameters()
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                self.gru.flatten_parameters()
                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

    def forward(self, inputs, rhs, done):
        """
        Forward pass Neural Network

        Parameters
        ----------
        inputs : tuple
            tuple with 3 positions. First, the obs, a tensor or a dict. Second,
            a tensor representing the recurrent hidden state. Finally, the
            current done tensor, indicating if episode has finished.

        Returns
        -------
        x : torch.tensor
            Output feature map.
        rhs : torch.tensor
            Updated recurrent hidden state.
        """

        x = inputs.view(inputs.size(0), -1)
        x, rhs = self._forward_gru(x, rhs, done)
        x = self.final_layer(x)
        x = self.final_activation(x)

        assert len(x.shape) == 2 and x.shape[1] == self.num_outputs

        return x, rhs


