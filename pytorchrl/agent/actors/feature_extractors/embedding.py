import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Embedding feature extractor.

    Parameters
    ----------
    vocabulary_size : int
        Number of tokens in the vocabulary.
    output_size : int
        Embedding layer size.
    """
    def __init__(self, input_space, vocabulary_size, output_size=256):
        super(Embedding, self).__init__()
        self._embedding = nn.Embedding(vocabulary_size, output_size)

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
        input_vector = torch.clamp(inputs, 0.0, self._embedding.num_embeddings).long()
        out = self._embedding(input_vector).squeeze(1)
        return out
