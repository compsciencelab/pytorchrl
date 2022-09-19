import gym
import numpy as np
import torch
import torch.nn as nn
from pytorchrl.agent.actors.utils import init
from transformers import OpenAIGPTConfig, OpenAIGPTModel


class GPT(nn.Module):
    """
    Wrapper to be able to use GPT model from transformers.

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    transformers_config : list
        Hidden layers sizes.
    """
    def __init__(self, input_space, transformers_config):
        super(GPT, self).__init__()

        # Define feature extractor
        self.feature_extractor = OpenAIGPTModel(transformers_config)

    def forward(self, inputs):
        """
        Forward pass Neural Network.

        This model assumes padding tokens are -1.

        Parameters
        ----------
        inputs : torch.tensor
            Input data.

        Returns
        -------
        out3 : torch.tensor
            Output feature map.
        """

        batch_size = inputs.size(0)
        inputs1 = inputs.view(batch_size, -1)
        inputs = inputs1 if len(inputs.shape) == 1 else inputs

        # Masks
        mask_attn = (inputs < 0.0)
        has_masked_tokens = (mask_attn == True).any()
        has_masked_tokens_in_all_sequences = mask_attn.any(dim=1).all() if batch_size > 1 else mask_attn.any()

        # Identify unnecessary padding
        if has_masked_tokens_in_all_sequences:
            ask_dim0, mask_dim1 = torch.where(mask_attn.all(dim=0, keepdims=True))
            last_idx = mask_dim1.min()
        else:
            last_idx = inputs.size(1) + 1

        # Remove -1 padding tokens
        inputs[mask_attn] = 0.0

        # Forward pass
        out1 = self.feature_extractor(
            input_ids=inputs1[:, 0:last_idx].long(),  # Shape (batch_size, sequence_length)
            attention_mask=1 - mask_attn[:, 0:last_idx].long(),  # Shape (batch_size, sequence_length)
        ).last_hidden_state

        # Prepare outputs
        if has_masked_tokens:  # Data collection
            idxs = (mask_attn.size(1) - mask_attn.sum(1)) - 1
            out2 = out1[torch.arange(len(out1)), idxs.view(-1)]
        else:  # Gradient computation
            out2 = out1[:, -1]

        # Place again padding tokens
        inputs[mask_attn] = - 1.0

        return out2
