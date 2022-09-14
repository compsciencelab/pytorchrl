"""
Implementation of the decorator using a Encoder-Decoder architecture.
"""
import math
import torch
import torch.nn as tnn
import torch.nn.utils.rnn as tnnur
from pytorchrl.agent.actors.feature_extractors import Embedding


class Encoder(tnn.Module):
    """
    Simple bidirectional RNN encoder implementation.
    """

    def __init__(self, num_layers, num_dimensions, vocabulary_size, dropout):
        super(Encoder, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.num_dimensions = num_dimensions
        self.vocabulary_size = vocabulary_size

        self._embedding = tnn.Sequential(
            tnn.Embedding(self.vocabulary_size, self.num_dimensions),
            tnn.Dropout(dropout)
        )
        self._rnn = tnn.LSTM(self.num_dimensions, self.num_dimensions, self.num_layers,
                             batch_first=True, dropout=self.dropout, bidirectional=True)

    def forward(self, padded_seqs, seq_lengths):  # pylint: disable=arguments-differ
        # FIXME: This fails with a batch of 1 because squeezing looses a dimension with size 1
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the sequences (batch, seq).
        :param seq_lengths: The lengths of the sequences (for packed sequences).
        :return : A tensor with all the output values for each step and the two hidden states.
        """

        import ipdb; ipdb.set_trace()

        batch_size = padded_seqs.size(0)
        max_seq_size = padded_seqs.size(1)
        hidden_state = self._initialize_hidden_state(batch_size)

        padded_seqs = self._embedding(padded_seqs.long())
        hs_h, hs_c = (hidden_state, hidden_state.clone().detach())

        # Is this faster?
        packed_seqs = tnnur.pack_padded_sequence(padded_seqs, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_seqs, (hs_h, hs_c) = self._rnn(packed_seqs, (hs_h, hs_c))
        padded_seqs, _ = tnnur.pad_packed_sequence(packed_seqs, batch_first=True)

        # padded_seqs, (hs_h, hs_c) = self._rnn(padded_seqs, (hs_h, hs_c))

        # sum up bidirectional layers and collapse
        hs_h = hs_h.view(self.num_layers, 2, batch_size, self.num_dimensions).sum(dim=1)  # .squeeze()  # (layers, batch, dim)
        hs_c = hs_c.view(self.num_layers, 2, batch_size, self.num_dimensions).sum(dim=1)  #.squeeze()  # (layers, batch, dim)
        padded_seqs = padded_seqs.view(batch_size, max_seq_size, 2, self.num_dimensions).sum(dim=2).squeeze(2)  # (batch, seq, dim)

        return padded_seqs, (hs_h, hs_c)

    def _initialize_hidden_state(self, batch_size):
        return torch.zeros(self.num_layers*2, batch_size, self.num_dimensions).cuda()

    def get_params(self):
        parameter_enums = GenerativeModelParametersEnum
        """
        Obtains the params for the network.
        :return : A dict with the params.
        """
        return {
            parameter_enums.NUMBER_OF_LAYERS: self.num_layers,
            parameter_enums.NUMBER_OF_DIMENSIONS: self.num_dimensions,
            parameter_enums.VOCABULARY_SIZE: self.vocabulary_size,
            parameter_enums.DROPOUT: self.dropout
        }


class AttentionLayer(tnn.Module):

    def __init__(self, num_dimensions):
        super(AttentionLayer, self).__init__()

        self.num_dimensions = num_dimensions

        self._attention_linear = tnn.Sequential(
            tnn.Linear(self.num_dimensions*2, self.num_dimensions),
            tnn.Tanh()
        )

    def forward(self, padded_seqs, encoder_padded_seqs, decoder_mask):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param encoder_padded_seqs: A tensor with the encoded input scaffold sequences (batch, seq_e, dim).
        :param decoder_mask: A tensor that represents the encoded input mask.
        :return : Two tensors: one with the modified logits and another with the attention weights.
        """

        # scaled dot-product
        # (batch, seq_d, 1, dim) * (batch, 1, seq_e, dim) => (batch, seq_d, seq_e*)
        attention_weights = (padded_seqs.unsqueeze(dim=2) * encoder_padded_seqs.unsqueeze(dim=1)) \
            .sum(dim=3).div(math.sqrt(self.num_dimensions)).softmax(dim=2)
        # (batch, seq_d, seq_e*)@(batch, seq_e, dim) => (batch, seq_d, dim)
        attention_context = attention_weights.bmm(encoder_padded_seqs)
        return (self._attention_linear(torch.cat([padded_seqs, attention_context], dim=2)) * decoder_mask, attention_weights)


class Decoder(tnn.Module):
    """
    Simple RNN decoder.
    """

    def __init__(self, num_layers, num_dimensions, vocabulary_size, dropout):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.num_dimensions = num_dimensions
        self.vocabulary_size = vocabulary_size
        self.dropout = dropout

        self._embedding = tnn.Sequential(
            tnn.Embedding(self.vocabulary_size, self.num_dimensions),
            tnn.Dropout(dropout)
        )
        self._rnn = tnn.LSTM(
            self.num_dimensions, self.num_dimensions, self.num_layers,
            batch_first=True, dropout=self.dropout, bidirectional=False)

        self._attention = AttentionLayer(self.num_dimensions)

    def forward(self, padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param seq_lengths: A list with the length of each output sequence.
        :param encoder_padded_seqs: A tensor with the encoded input scaffold sequences (batch, seq_e, dim).
        :param hidden_states: The hidden states from the encoder.
        :return : Three tensors: The output logits, the hidden states of the decoder and the attention weights.
        """

        padded_encoded_seqs = self._embedding(padded_seqs.long())

        # Is it faster ?
        packed_encoded_seqs = tnnur.pack_padded_sequence(padded_encoded_seqs, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_encoded_seqs, hidden_states = self._rnn(packed_encoded_seqs, hidden_states)
        padded_encoded_seqs, _ = tnnur.pad_packed_sequence(packed_encoded_seqs, batch_first=True)  # (batch, seq, dim)

        # padded_encoded_seqs, hidden_states = self._rnn(padded_encoded_seqs, hidden_states)

        # import ipdb; ipdb.set_trace() # What is the mask?
        mask = (padded_encoded_seqs[:, :, 0] != 0).unsqueeze(dim=-1).type(torch.float)
        attn_padded_encoded_seqs, attention_weights = self._attention(padded_encoded_seqs, encoder_padded_seqs, mask)

        logits = attn_padded_encoded_seqs

        return logits, hidden_states, attention_weights

    def get_params(self):
        parameter_enum = GenerativeModelParametersEnum
        """
        Obtains the params for the network.
        :return : A dict with the params.
        """
        return {
            parameter_enum.NUMBER_OF_LAYERS: self.num_layers,
            parameter_enum.NUMBER_OF_DIMENSIONS: self.num_dimensions,
            parameter_enum.VOCABULARY_SIZE: self.vocabulary_size,
            parameter_enum.DROPOUT: self.dropout
        }


class Decorator(tnn.Module):
    """
    An encoder-decoder that decorates scaffolds.
    """

    def __init__(self, input_size, encoder_params, decoder_params):
        super(Decorator, self).__init__()

        encoder_params.update({
            "num_layers": 3,
            "num_dimensions": 512,
            "dropout": 0,
        })

        decoder_params.update({
            "num_layers": 3,
            "num_dimensions": 512,
            "dropout": 0,
        })

        self._encoder = Encoder(**encoder_params)
        self._decoder = Decoder(**decoder_params)
        self.encoder_rhs = None
        self.encoder_padded_seqs = None

    def _forward_decorator(self, x, hxs, done):

        encoder_seqs = x["scaffold"]
        decoder_seqs = x["decoration"]
        encoder_seq_lengths = x["scaffold_length"].cpu().long()
        decoder_seq_lengths = x["decoration_length"].cpu().long()

        masks = 1 - done

        if decoder_seqs.size(0) == hxs.size(0):

            #### LSTM CODE ########

            # self._rnn.flatten_parameters()
            # x, hxs = self._rnn(x.unsqueeze(0), torch.chunk((torch.transpose(hxs, 0, 1) * masks).contiguous(), 2))
            # hxs = torch.transpose(torch.cat(hxs), 0, 1)
            # x = x.squeeze(0)

            ########################

            # if self.encoder_rhs is None or self.encoder_padded_seqs is None:
            self.encoder_padded_seqs, self.encoder_rhs = self.forward_encoder(encoder_seqs, encoder_seq_lengths)
            self.encoder_rhs = torch.transpose(torch.cat(self.encoder_rhs), 0, 1)

            # Replace "done" hxs by self.encoder_rhs
            # hxs = torch.where(done.unsqueeze(-1) > 0.0, self.encoder_rhs, hxs)

            # Chunk rhs (where does this go?)
            hxs = torch.chunk((torch.transpose(hxs, 0, 1)), 2)

            logits, hxs, _ = self.forward_decoder(decoder_seqs, decoder_seq_lengths, self.encoder_padded_seqs, hxs)

            logits = logits.squeeze(1)
            hxs = torch.transpose(torch.cat(hxs), 0, 1)

        else:

            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(decoder_seqs.size(0) / N)

            # # Set encoder outputs to None
            # self.encoder_rhs = None
            # self.encoder_padded_seqs = None
            #
            # # unflatten
            # decoder_seqs = decoder_seqs.view(N, T, -1)
            #
            # # Same deal with masks
            # masks = masks.view(N, T)
            #
            # # Let's figure out which steps in the sequence have a zero for any agent
            # # We will always assume t=0 has a zero in it as that makes the logic cleaner
            # # has_zeros_old = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())
            # has_zeros = torch.nonzero(((masks[1:] == 0.0).any(dim=-1)), as_tuple=False).squeeze().cpu()
            # # assert (has_zeros_old == has_zeros).all()
            #
            # # +1 to correct the masks[1:]
            # if has_zeros.dim() == 0:
            #     # Deal with scalar
            #     has_zeros = [has_zeros.item() + 1]
            # else:
            #     has_zeros = (has_zeros + 1).numpy().tolist()
            #
            # # add t=0 and t=T to the list
            # has_zeros = [0] + has_zeros + [T]
            #
            # outputs = []
            # for i in range(len(has_zeros) - 1):
            #     # We can now process steps that don't have any zeros in masks together!
            #     # This is much faster
            #     start_idx = has_zeros[i]
            #     end_idx = has_zeros[i + 1]
            #
            #     # TODO: run encoder in position start_idx if required
            #     if self.encoder_rhs is None or self.encoder_padded_seqs is None:
            #         lengths = encoder_seq_lengths[start_idx: start_idx + 1].cpu().long()
            #         if lengths == 0: lengths += encoder_seqs.size(1)
            #         self.encoder_padded_seqs, self.encoder_rhs = self.forward_encoder(
            #             encoder_seqs[start_idx: start_idx + 1], lengths)
            #
            #     # TODO: run decoder from start_idx to end_idx
            #     lengths = torch.LongTensor([end_idx - start_idx])
            #     logits, hxs, _ = self.forward_decoder(decoder_seqs[:, start_idx:end_idx].squeeze(-1), lengths, self.encoder_padded_seqs, self.encoder_rhs)
            #
            #     outputs.append(logits)
            #
            # # x is a (T, N, -1) tensor
            # logits = torch.cat(outputs, dim=0)
            #
            # # flatten
            # logits = logits.view(T * N, -1)
            #
            # hxs = torch.transpose(torch.cat(hxs), 0, 1)
            #
            # # Set encoder outputs to None
            # self.encoder_rhs = None
            # self.encoder_padded_seqs = None

            logits = torch.zeros(T * N, self._encoder.num_dimensions).cuda()

        return logits, hxs

    def forward(self, inputs, rhs, done):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param encoder_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param encoder_seq_lengths: A list with the length of each input sequence.
        :param decoder_seqs: A tensor with the encoded input scaffold sequences (batch, seq_e, dim).
        :param decoder_seq_lengths: The lengths of the decoder sequences.
        :return : The output logits as a tensor (batch, seq_d, dim).
        """

        logits, rhs = self._forward_decorator(inputs, rhs, done)

        return logits, rhs

    def forward_encoder(self, padded_seqs, seq_lengths):
        """
        Does a forward pass only of the encoder.
        :param padded_seqs: The data to feed the encoder.
        :param seq_lengths: The length of each sequence in the batch.
        :return : Returns a tuple with (encoded_seqs, hidden_states)
        """
        return self._encoder(padded_seqs, seq_lengths)

    def forward_decoder(self, padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states):
        """
        Does a forward pass only of the decoder.
        :param hidden_states: The hidden states from the encoder.
        :param padded_seqs: The data to feed to the decoder.
        :param seq_lengths: The length of each sequence in the batch.
        :return : Returns the logits and the hidden state for each element of the sequence passed.
        """
        return self._decoder(padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states)

    @property
    def num_outputs(self):
        """Output feature map size (as in np.prod(self.output_shape))."""
        return self._decoder.num_dimensions

    @property
    def recurrent_hidden_state_size(self):
        """Recurrent hidden state size"""
        return self._decoder.num_dimensions

    def get_initial_recurrent_state(self, num_proc):
        return torch.zeros(num_proc, self._encoder.num_layers * 2, self._encoder.num_dimensions)
