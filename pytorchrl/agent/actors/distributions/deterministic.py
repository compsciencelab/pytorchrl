import torch
import torch.nn as nn

from pytorchrl.agent.actors.utils import init
from pytorchrl.agent.actors.noise import get_noise


class Deterministic(nn.Module):
    """
    Deterministic prediction of the mean value mu of a learned action distribtion.

    Parameters
    ----------
    num_inputs : int
        Size of input feature maps.
    num_outputs : int
        Number of dims in output space.
    noise : str
        Type of noise that is added to the predicted mu.
    """

    def __init__(self, num_inputs, num_outputs, noise):
        super(Deterministic, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.action_output = init_(nn.Linear(num_inputs, num_outputs))
        self.noise = get_noise(noise)(num_outputs)

    def forward(self, x, deterministic=True):
        """
        Predict distribution parameters from x (obs features)
        and returns predicted noisy action mu of the distribution and the clipped action [-1, 1].

        Parameters
        ----------
        x : torch.tensor
            Feature maps extracted from environment observations.
        deterministic : bool
            Whether to noise is added to the predicted mu or not.
            
        Returns
        -------
        action: torch.tensor
            Next action sampled.
        clipped_action: torch.tensor
            Next action sampled, but clipped to be within the env action space.
        logp: None
            Returns logp 'None' to have equal output to other distributions.
        entropy_dist: None
            Returns logp 'None' to have equal output to other distributions
        """
        mu = torch.tanh(self.action_output(x))
        if not deterministic:
            noise = self.noise.sample().to(mu.device)
            mu = mu + noise
        clipped_action = torch.clamp(mu, min=-1, max=1)
            
        return mu, clipped_action, None, None


    def evaluate_pred(self, x):
        """
        Predict distribution parameters from x (obs features)
        and returns predicted mu value of the distribution.

        Parameters
        ----------
        x : torch.tensor
            Feature maps extracted from environment observations.

        Returns
        -------
        pred: torch.tensor
            Predicted value.
        """
        return torch.tanh(self.action_output(x))
