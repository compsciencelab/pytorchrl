"""
Implementation of the RNN model.

Adapted from https://github.com/MolecularAI/reinvent-models/blob/main/reinvent_models/reinvent_core/models/model.py
"""
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.reinvent_core.models import vocabulary as mv

from pytorchrl.agent.actors.distributions import get_dist


class RNN(nn.Module):
    """
    Implements a N layer GRU(M) cell including an embedding layer
    and an output linear layer back to the size of the vocabulary
    """

    def __init__(self, voc_size, layer_size=512, num_layers=3, cell_type='gru', embedding_layer_size=256, dropout=0.,
                 layer_normalization=False):
        """
        Implements a N layer GRU|LSTM cell including an embedding layer and an output linear layer back to the size of the
        vocabulary
        :param voc_size: Size of the vocabulary.
        :param layer_size: Size of each of the RNN layers.
        :param num_layers: Number of RNN layers.
        :param embedding_layer_size: Size of the embedding layer.
        """
        super(RNN, self).__init__()

        self._layer_size = layer_size
        self._embedding_layer_size = embedding_layer_size
        self._num_layers = num_layers
        self._cell_type = cell_type.lower()
        self._dropout = dropout
        self._layer_normalization = layer_normalization

        self._embedding = nn.Embedding(voc_size, self._embedding_layer_size)
        if self._cell_type == 'gru':
            self._rnn = nn.GRU(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                               dropout=self._dropout, batch_first=False)
        elif self._cell_type == 'lstm':
            self._rnn = nn.LSTM(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                                dropout=self._dropout, batch_first=False)
        else:
            raise ValueError('Value of the parameter cell_type should be "gru" or "lstm"')

    def forward(self, input_vector, hidden_state=None, done=False):
        """
        Performs a forward pass on the model. Note: you pass the **whole** sequence.
        :param input_vector: Input tensor (batch_size, seq_size).
        :param hidden_state: Hidden state tensor.
        """
        input_vector = torch.clamp(input_vector, 0.0, self._embedding.num_embeddings).long()
        batch_size, seq_size = input_vector.size()
        embedded_data = self._embedding(input_vector).squeeze(1)  # (batch, seq, embedding)
        output_vector, hidden_state_out = self._forward_memory_net(embedded_data, hidden_state, done)

        if self._layer_normalization:
            output_vector = nnf.layer_norm(output_vector, output_vector.size()[1:])

        # output_vector = output_vector.reshape(-1, self._layer_size)

        return output_vector, hidden_state_out

    def _forward_memory_net(self, x, hxs, done):
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
            if self._cell_type == "gru":
                x, hxs = self._rnn(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
                x = x.squeeze(0)
                hxs = hxs.squeeze(0)
            else:
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

            if self._cell_type == "gru":
                hxs = hxs.unsqueeze(0)

            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                self._rnn.flatten_parameters()

                if self._cell_type == "gru":
                    rnn_scores, hxs = self._rnn(
                        x[start_idx:end_idx],
                        hxs * masks[start_idx].view(1, -1, 1))
                else:

                    rnn_scores, hxs = self._rnn(
                        x[start_idx:end_idx],
                        torch.chunk(
                            (torch.transpose(hxs, 0, 1) * masks[start_idx].view(1, -1, 1)).contiguous(),
                            2))
                    hxs = torch.transpose(torch.cat(hxs), 0, 1)

                outputs.append(rnn_scores)

            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

            if self._cell_type == "gru":
                hxs = hxs.squeeze(0)

        return x, hxs

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        """
        return {
            'dropout': self._dropout,
            'layer_size': self._layer_size,
            'num_layers': self._num_layers,
            'cell_type': self._cell_type,
            'embedding_layer_size': self._embedding_layer_size
        }

    def get_initial_recurrent_state(self):
        if self._cell_type == 'gru':
            self._rnn = nn.GRU(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                               dropout=self._dropout, batch_first=True)
        elif self._cell_type == 'lstm':
            self._rnn = nn.LSTM(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                                dropout=self._dropout, batch_first=True)
        else:
            raise ValueError('Value of the parameter cell_type should be "gru" or "lstm"')
        return


class Seq2Seq(nn.Module):
    """
    Implements an RNN model using SMILES.
    """

    def __init__(self,
                 input_space,
                 vocabulary: mv.Vocabulary,
                 tokenizer, layer_size=512, num_layers=3, cell_type='gru', embedding_layer_size=256, dropout=0.,
                 layer_normalization=False, max_sequence_length=256):
        """
        Implements an RNN.
        :param vocabulary: Vocabulary to use.
        :param tokenizer: Tokenizer to use.
        :param network_params: Dictionary with all parameters required to correctly initialize the RNN class.
        :param max_sequence_length: The max size of SMILES sequence that can be generated.
        """

        super(Seq2Seq, self).__init__()

        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self._model_modes = ModelModeEnum()
        self._num_outputs = layer_size
        self.network = RNN(
            dropout=dropout,
            cell_type=cell_type,
            num_layers=num_layers,
            layer_size=layer_size,
            voc_size=len(self.vocabulary),
            layer_normalization=layer_normalization,
            embedding_layer_size=embedding_layer_size)

    @property
    def num_outputs(self):
        """Output feature map size (as in np.prod(self.output_shape))."""
        return self._num_outputs

    @property
    def recurrent_hidden_state_size(self):
        """Recurrent hidden state size"""
        return self._num_outputs

    def forward(self, inputs, rhs, done):
        logits, hidden_state = self.network(inputs, rhs, done)
        return logits, hidden_state

    def set_mode(self, mode: str):
        if mode == self._model_modes.TRAINING:
            self.network.train()
        elif mode == self._model_modes.INFERENCE:
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")
