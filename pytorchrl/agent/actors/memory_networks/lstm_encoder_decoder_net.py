"""
Implementation of the decorator using a Encoder-Decoder architecture.
Adapted from https://github.com/MolecularAI/reinvent-models.
"""
import math
import torch
import torch.nn as nn
import torch.nn.utils.rnn as tnnur
from pytorchrl.agent.actors.feature_extractors import Embedding


class Encoder(nn.Module):
    """LSTM Bidirectional RNN encoder model."""

    def __init__(self, vocabulary_size, num_layers=3, num_dimensions=512, dropout=0.0):
        """
        Initializes the LSTM Bidirectional RNN encoder model.

        Parameters
        ----------
        vocabulary_size : int
            Number of possible tokens in the input space.
        num_dimensions : int
            Recurrent hidden state and output size.
        dropout : float
            If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer.
        num_layers : int
            Number of recurrent layers.
        """

        super(Encoder, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.num_dimensions = num_dimensions
        self.vocabulary_size = vocabulary_size

        self._embedding = nn.Sequential(
            nn.Embedding(self.vocabulary_size, self.num_dimensions),
            nn.Dropout(dropout))

        self._rnn = nn.LSTM(
            self.num_dimensions, self.num_dimensions, self.num_layers,
            batch_first=True, dropout=self.dropout, bidirectional=True)

    def forward(self, padded_seqs, seq_lengths):
        """
        Performs encoder forward pass.

        Parameters
        ----------
        padded_seqs : torch.tensor
            A tensor with the sequences (batch, seq).
        seq_lengths : torch.tensor
            1D tensor with the lengths of the sequences (for packed sequences).

        Returns
        -------
        padded_seqs : torch.tensor
            LSTM outputs values.
        (hs_h, hs_c) : tuple
            LSTM last recurrent hidden state.
        """

        batch_size = padded_seqs.size(0)
        hidden_state = self._initialize_hidden_state(batch_size)

        padded_seqs = self._embedding(padded_seqs.long())
        hs_h, hs_c = (hidden_state, hidden_state.clone().detach())

        packed_seqs = tnnur.pack_padded_sequence(padded_seqs, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_seqs, (hs_h, hs_c) = self._rnn(packed_seqs, (hs_h, hs_c))
        padded_seqs, _ = tnnur.pad_packed_sequence(packed_seqs, batch_first=True)

        # sum up bidirectional layers and collapse
        hs_h = hs_h.view(self.num_layers, 2, batch_size, self.num_dimensions).sum(dim=1)
        hs_c = hs_c.view(self.num_layers, 2, batch_size, self.num_dimensions).sum(dim=1)
        padded_seqs = padded_seqs.view(batch_size, -1, 2, self.num_dimensions).sum(dim=2).squeeze(2)  # (batch, seq, dim)

        return padded_seqs, (hs_h, hs_c)

    def _initialize_hidden_state(self, batch_size):
        """Returns a tensor of zeros with the expected shape of the model's rhs."""
        return torch.zeros(self.num_layers*2, batch_size, self.num_dimensions).cuda()


class AttentionLayer(nn.Module):

    def __init__(self, num_dimensions):
        super(AttentionLayer, self).__init__()

        self.num_dimensions = num_dimensions

        self._attention_linear = nn.Sequential(
            nn.Linear(self.num_dimensions*2, self.num_dimensions), nn.Tanh())

    def forward(self, padded_seqs, encoder_padded_seqs, decoder_mask):
        """Performs the forward pass.

        Parameters
        ----------
        padded_seqs : torch.tensor
            A tensor with the decoder output sequences (batch, seq_d, dim).
        encoder_padded_seqs : torch.tensor
             tensor with the encoded context sequences (batch, seq_e, dim).
        decoder_mask : torch.tensor
            A tensor that represents the encoded input mask.

        Returns
        -------
        logits : torch.tensor
            Modified logits.
        attention_weights : torch.tensor
            Model attention weights.
        """

        # scaled dot-product
        # (batch, seq_d, 1, dim) * (batch, 1, seq_e, dim) => (batch, seq_d, seq_e)
        attention_weights = (padded_seqs.unsqueeze(dim=2) * encoder_padded_seqs.unsqueeze(dim=1)).sum(
            dim=3).div(math.sqrt(self.num_dimensions)).softmax(dim=2)

        # (batch, seq_d, seq_e) @ (batch, seq_e, dim) => (batch, seq_d, dim)
        attention_context = attention_weights.bmm(encoder_padded_seqs)
        return (self._attention_linear(torch.cat(
            [padded_seqs, attention_context], dim=2)) * decoder_mask, attention_weights)


class Decoder(nn.Module):
    """LSTM RNN decoder model."""

    def __init__(self, vocabulary_size, num_layers=3, num_dimensions=512, dropout=0.0):
        """
        Initializes the LSTM decoder model.

        Parameters
        ----------
        vocabulary_size : int
            Number of possible tokens in the input space.
        num_dimensions : int
            Recurrent hidden state and output size.
        dropout : float
            If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer.
        num_layers : int
            Number of recurrent layers.
        """
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.num_dimensions = num_dimensions
        self.vocabulary_size = vocabulary_size
        self.dropout = dropout

        self._embedding = nn.Sequential(
            nn.Embedding(self.vocabulary_size, self.num_dimensions),
            nn.Dropout(dropout))

        self._rnn = nn.LSTM(
            self.num_dimensions, self.num_dimensions, self.num_layers,
            batch_first=True, dropout=self.dropout, bidirectional=False)

        self._attention = AttentionLayer(self.num_dimensions)

    def forward(self, padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states):
        """
        Performs decoder forward pass.

        Parameters
        ----------
        padded_seqs : torch.tensor
            A tensor with the sequences (batch, seq).
        seq_lengths : torch.tensor
            1D tensor with the lengths of the sequences (for packed sequences).
        encoder_padded_seqs : torch.tensor
             tensor with the encoded context sequences (batch, seq_e, dim).
        hidden_states : tuple
            Last recurrent hidden states.

        Returns
        -------
        logits : torch.tensor
            Output logits.
        hidden_states : tuple
            Updated recurrent hidden state.
        attention_weights : tensor
            Attention layer weights.
        """

        padded_encoded_seqs = self._embedding(padded_seqs.long())

        packed_encoded_seqs = tnnur.pack_padded_sequence(padded_encoded_seqs, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_encoded_seqs, hidden_states = self._rnn(packed_encoded_seqs, hidden_states)
        padded_encoded_seqs, _ = tnnur.pad_packed_sequence(packed_encoded_seqs, batch_first=True)  # (batch, seq, dim)

        mask = (padded_encoded_seqs[:, :, 0] != 0).unsqueeze(dim=-1).type(torch.float)
        attn_padded_encoded_seqs, attention_weights = self._attention(padded_encoded_seqs, encoder_padded_seqs, mask)
        logits = attn_padded_encoded_seqs * mask

        return logits, hidden_states, attention_weights


class LSTMEncoderDecoder(nn.Module):
    """
    An encoder-decoder model that encodes context and then uses a decoder and an attention layer
    to autoregressively generate actions.

    Action space is expected to be a dict, containing at least the following fields:
        - context: padded context sequences, expected shape=(batch_size, sequences_length)
        - context_length: a 1D tensor with the length of each context sequence.
        - obs: next episode observations, shape=(batch_size, **obs_input_shape)
        - obs_length: a 1D tensor with the length of each obs sequence (can be 1).

    Parameters
    ----------
    input_size : int
        Input feature map size (unused).
    encoder_params : dict
        Dictionary specifying encoder's vocabulary_size, num_layers (default 3), num_dimensions (default 512) and
        dropout (default 0.0).
    decoder_params : dict
        Dictionary specifying decoder's vocabulary_size, num_layers (default 3), num_dimensions (default 512) and
        dropout (default 0.0).
    """

    def __init__(self, input_size, encoder_params={}, decoder_params={}):
        super(LSTMEncoderDecoder, self).__init__()

        self._encoder = Encoder(**encoder_params)
        self._decoder = Decoder(**decoder_params)
        self.encoder_rhs = None
        self.encoder_padded_seqs = None

    def _forward_encoder(self, padded_seqs, seq_lengths):
        """
        Does a forward pass only of the encoder.

        Parameters
        ----------
        padded_seqs : torch.tensor
            Input data to feed the encoder.
        seq_lengths : torch.tensor
            1D tensor with the lengths of each padded_seqs sequences (for packed sequences).

        Returns
        -------
        encoded_seqs : torch.tensor
            Output encoded sequences.
        hidden_states : tuple
            Updated recurrent hidden state.
        """
        return self._encoder(padded_seqs, seq_lengths)

    def _forward_decoder(self, padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states):
        """
        Does a forward pass only of the decoder.

        Parameters
        ----------
        padded_seqs : torch.tensor
            Input data to feed the decoder.
        seq_lengths : torch.tensor
            1D tensor with the lengths of each padded_seqs sequences (for packed sequences).
        hidden_states : tuple
            Last recurrent hidden states.

        Returns
        -------
        logits : torch.tensor
            Output decoder logits.
        hidden_states : tuple
            Updated recurrent hidden state.
        """
        return self._decoder(padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states)

    def _forward_encoder_decoder(self, encoder_seqs, encoder_seq_lengths, decoder_seqs, decoder_seq_lengths, hxs, done):

        masks = 1 - done

        if decoder_seqs.size(0) == hxs.size(0):

            if self.encoder_rhs is None or self.encoder_padded_seqs is None:
                self.encoder_padded_seqs, self.encoder_rhs = self._forward_encoder(encoder_seqs, encoder_seq_lengths)
                self.encoder_rhs = torch.transpose(torch.cat(self.encoder_rhs), 0, 1)

            # Replace "done" hxs by self.encoder_rhs
            hxs = torch.where(done.unsqueeze(-1) > 0.0, self.encoder_rhs, hxs)

            # Chunk rhs (where does this go?)
            hxs = torch.chunk((torch.transpose(hxs, 0, 1)), 2)

            logits, hxs, _ = self._forward_decoder(decoder_seqs, decoder_seq_lengths, self.encoder_padded_seqs, hxs)
            logits = logits.squeeze(1)

            hxs = torch.transpose(torch.cat(hxs), 0, 1)

        else:

            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(decoder_seqs.size(0) / N)

            # Set encoder outputs to None
            self.encoder_rhs = None
            self.encoder_padded_seqs = None

            # unflatten
            decoder_seqs = torch.transpose(decoder_seqs.view(T, N, -1), 0, 1)
            encoder_seqs = encoder_seqs.view(T, N, -1)[0]
            encoder_seq_lengths[encoder_seq_lengths == 0] = encoder_seqs.size(1)
            encoder_seq_lengths = encoder_seq_lengths.view(T, N)[0]

            # Same deal with masks
            masks = torch.transpose(masks.view(T, N), 0, 1)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = torch.nonzero(((masks[:, 1:] == 0.0).any(dim=0)), as_tuple=False).squeeze().cpu()

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

                # TODO: run encoder in position start_idx if required
                if self.encoder_rhs is None or self.encoder_padded_seqs is None:
                    self.encoder_padded_seqs, self.encoder_rhs = self._forward_encoder(
                        encoder_seqs, encoder_seq_lengths)
                    self.encoder_rhs = torch.transpose(torch.cat(self.encoder_rhs), 0, 1)

                hxs = torch.where(masks[:, start_idx: start_idx + 1].unsqueeze(-1) == 0.0, self.encoder_rhs, hxs)

                # Chunk rhs (where does this go?)
                hxs = torch.chunk((torch.transpose(hxs, 0, 1)), 2)

                # Run decoder from start_idx to end_idx
                lengths = torch.cat([torch.LongTensor([end_idx - start_idx])] * N)
                logits, hxs, _ = self._forward_decoder(
                    decoder_seqs[:, start_idx:end_idx].squeeze(-1), lengths, self.encoder_padded_seqs, hxs)

                hxs = torch.transpose(torch.cat(hxs), 0, 1)

                outputs.append(logits)

            # x is a (T, N, -1) tensor
            logits = torch.cat(outputs, dim=1)

            # flatten
            logits = torch.transpose(logits, 0, 1).reshape(T * N, -1)

            # Set encoder outputs to None
            self.encoder_rhs = None
            self.encoder_padded_seqs = None

        return logits, hxs

    def forward(self, inputs, rhs, done):
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

        encoder_seqs = inputs["context"]
        decoder_seqs = inputs["obs"]
        encoder_seq_lengths = inputs["context_length"].cpu().long()
        decoder_seq_lengths = inputs["obs_length"].cpu().long()

        logits, rhs = self._forward_encoder_decoder(
            encoder_seqs, encoder_seq_lengths, decoder_seqs, decoder_seq_lengths, rhs, done)

        return logits, rhs

    @property
    def num_outputs(self):
        """Output feature map size (as in np.prod(self.output_shape))."""
        return self._decoder.num_dimensions

    @property
    def recurrent_hidden_state_size(self):
        """Recurrent hidden state size"""
        return self._decoder.num_dimensions

    def get_initial_recurrent_state(self, num_proc):
        """Returns a tensor of zeros with the expected shape of the model's rhs."""
        return torch.zeros(num_proc, self._encoder.num_layers * 2, self._encoder.num_dimensions)
