from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class LstmNet(nn.Module):
    """Implements a LSTM model."""

    def __init__(self, input_size, layer_size=512, num_layers=3, dropout=0., layer_normalization=False):
        """
        Initializes a N layer LSTM cell.

        Parameters
        ----------
        input_size : int
            Input feature map size.
        output_size : int
            Recurrent hidden state and output size.
        dropout : float
            If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer.
        num_layers : int
            Number of recurrent layers.
        activation : func
            Non-linear activation function.
        layer_normalization : bool
            If True, adds a layer normalization module at the end of the model.
        """
        super(LstmNet, self).__init__()
        self._input_size = input_size
        self._layer_size = layer_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._layer_normalization = layer_normalization
        self._rnn = nn.LSTM(
            self._input_size, self._layer_size, num_layers=self._num_layers,
            dropout=self._dropout, batch_first=False)

    @property
    def num_outputs(self):
        """Output feature map size (as in np.prod(self.output_shape))."""
        return self._layer_size

    @property
    def recurrent_hidden_state_size(self):
        """Recurrent hidden state size"""
        return self._layer_size

    def _forward_lstm(self, x, hxs, done):
        """
        Fast forward pass memory network.

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
            self._rnn.flatten_parameters()
            x, hxs = self._rnn(x.unsqueeze(0), torch.chunk((torch.transpose(hxs, 0, 1) * masks).contiguous(), 2))
            hxs = torch.transpose(torch.cat(hxs), 0, 1)
            x = x.squeeze(0)

        else:

            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, -1)

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = torch.nonzero(((masks[1:] == 0.0).any(dim=-1)), as_tuple=False).squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                self._rnn.flatten_parameters()
                rnn_scores, hxs = self._rnn(
                    x[start_idx:end_idx],
                    torch.chunk((torch.transpose(hxs, 0, 1) * masks[start_idx].view(
                        1, -1, 1)).contiguous(), 2))
                hxs = torch.transpose(torch.cat(hxs), 0, 1)
                outputs.append(rnn_scores)

            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.view(T * N, -1)

        return x, hxs

    def forward(self, inputs, rhs=None, done=False):
        """
        Forward pass Neural Network

        Parameters
        ----------
        inputs : torch.tensor
            A tensor containing episode observations.
        rhs : torch.tensor
            A tensor representing the recurrent hidden states.
        done : torch.tensor
            A tensor indicating where episodes end.

        Returns
        -------
        x : torch.tensor
            Output feature map.
        rhs : torch.tensor
            Updated recurrent hidden state.
        """
        x = inputs.view(inputs.size(0), -1)
        x, rhs = self._forward_lstm(x, rhs, done)

        if self._layer_normalization:
            x = nnf.layer_norm(x, x.size()[1:])

        return x, rhs

    def get_initial_recurrent_state(self, num_proc):
        """Returns a tensor of zeros with the expected shape of the model's rhs."""
        return torch.zeros(num_proc, self._num_layers * 2, self._layer_size)
