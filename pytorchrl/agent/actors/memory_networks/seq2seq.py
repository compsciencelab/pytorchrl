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
                               dropout=self._dropout, batch_first=True)
        elif self._cell_type == 'lstm':
            self._rnn = nn.LSTM(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                                dropout=self._dropout, batch_first=True)
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
        embedded_data = self._embedding(input_vector)  # (batch, seq, embedding)
        size = (self._num_layers, batch_size, self._layer_size)

        if self._cell_type == "gru":

            if hidden_state.sum() == 0.0:
                hidden_state = torch.zeros(*size).to(input_vector.device)
            output_vector, hidden_state_out = self._rnn(embedded_data, hidden_state)

        else:

            if hidden_state.sum() == 0.0:
                hidden_state = [torch.zeros(*size).to(input_vector.device), torch.zeros(*size).to(input_vector.device)]
                hidden_state = torch.cat(hidden_state)
            hidden_state = torch.chunk(hidden_state, 2)
            output_vector, hidden_state_out = self._rnn(embedded_data, hidden_state)
            hidden_state_out = torch.cat(hidden_state_out)

        if self._layer_normalization:
            output_vector = nnf.layer_norm(output_vector, output_vector.size()[1:])

        # output_vector = output_vector.reshape(-1, self._layer_size)
        output_vector = output_vector[:, -1, :]

        return output_vector, hidden_state_out

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
