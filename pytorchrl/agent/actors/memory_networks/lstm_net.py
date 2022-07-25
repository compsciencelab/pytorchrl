from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class LstmNet(nn.Module):
    """
    Implements a N layer GRU(M) cell including an embedding layer
    and an output linear layer back to the size of the vocabulary
    """

    def __init__(self, input_size, layer_size=512, num_layers=3, dropout=0., layer_normalization=False):
        """Implements a N layer LSTM cell."""
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

    def forward(self, input_vector, hidden_state=None, done=False):
        """
        Performs a forward pass on the model. Note: you pass the **whole** sequence.
        :param input_vector: Input tensor (batch_size, seq_size).
        :param hidden_state: Hidden state tensor.
        """
        input_vector = input_vector.view(input_vector.size(0), -1)
        output_vector, hidden_state_out = self._forward_lstm(input_vector, hidden_state, done)

        if self._layer_normalization:
            output_vector = nnf.layer_norm(output_vector, output_vector.size()[1:])

        return output_vector, hidden_state_out

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
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

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

    def get_initial_recurrent_state(self, num_proc):
        return torch.zeros(num_proc, self._num_layers * 2, self._layer_size)
