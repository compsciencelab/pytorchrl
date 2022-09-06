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

        inputs = inputs.view(inputs.size(0), -1)
        inputs = {
            'input_ids': inputs.long(),  # Shape (batch_size, sequence_length)
            'attention_mask': torch.ones_like(inputs).long(),  # Shape (batch_size, sequence_length)
        }
        out = self.feature_extractor(**inputs)
        return out.last_hidden_state
