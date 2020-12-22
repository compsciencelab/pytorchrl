import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from ..neural_networks.feature_extractors.utils import init


LOG_STD_MAX = 2 # Maximum std allowed value. Used for clipping.
LOG_STD_MIN = -20 # Minimum std allowed value. Used for clipping.

class SquashedGaussian(nn.Module):
    """
    Squashed Gaussian action distribution.

    Parameters
    ----------
    num_inputs : int
        Size of input feature maps.
    num_outputs : int
        Number of action in action space.
    predict_log_std : bool
        Whether to use a nn.linerar layer to predict the output actions std.

    Attributes
    ----------
    mean: nn.Module
        Maps the incoming feature maps to output action mean values.
    log_std : nn.Module or nn.Parameter
        If predict_log_std is True Maps the incoming feature maps to output
        action std values. Otherwise, the std values are a learnable nn.Parameter.
    """

    def __init__(self, num_inputs, num_outputs, predict_log_std=True):
        super(SquashedGaussian, self).__init__()

        self.predict_log_std = predict_log_std
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.mean = init_(nn.Linear(num_inputs, num_outputs))

        if predict_log_std:
            self.log_std = init_(nn.Linear(num_inputs, num_outputs))
        else:
            self.log_std = nn.Parameter(torch.zeros(num_outputs).unsqueeze(0))

    def forward(self, x, deterministic=False):
        """
        Predict distribution parameters from x (obs features) and return
        actions (detemrninistic, sampled and clipped), sampled action log
        probability and distribution entropy.

        Parameters
        ----------
        x : torch.tensor
            Feature maps extracted from environment observations.
        deterministic : bool
            Whether to randomly sample action from predicted distribution or take the mode.

        Returns
        -------
        action: torch.tensor
            Predicted next action.
        clipped_action: torch.tensor
            Predicted next action (clipped to be within action space).
        logp_action : torch.tensor
            Log probability of `action` according to the predicted action distribution.
        entropy_dist : torch.tensor
            Entropy of the predicted action distribution.
        """

        # Predict distribution parameters
        action_mean = self.mean(x)
        action_logstd = self.log_std(x) if self.predict_log_std else\
            torch.zeros(action_mean.size()).to(x.device) + self.log_std
        action_logstd = torch.clamp(action_logstd, LOG_STD_MIN, LOG_STD_MAX)

        # Create distribution and sample action
        action_dist = Normal(action_mean, action_logstd.exp())

        if deterministic:
            action = action_mean
        else:
            action = action_dist.rsample()

        # Action log probability
        logp_action = action_dist.log_prob(action).sum(axis=-1, keepdim=True)
        logp_action -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
            axis=1, keepdim=True)

        # Apply squashing
        action = clipped_action = torch.tanh(action)

        # Distribution entropy
        entropy_dist = -logp_action.mean()

        return action, clipped_action, logp_action, entropy_dist
