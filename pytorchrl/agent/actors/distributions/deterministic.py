import torch
import torch.nn as nn

from pytorchrl.agent.actors.utils import init
from pytorchrl.agent.actors.noise import get_noise
from pytorchrl.agent.actors.feature_extractors.ensemble_layer import EnsembleFC


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
        dist : torch.Distribution
            Action probability distribution.
        """
        mu = torch.tanh(self.action_output(x))
        if not deterministic:
            noise = self.noise.sample().to(mu.device)
            mu = mu + noise
        clipped_action = torch.clamp(mu, min=-1, max=1)
            
        return mu, clipped_action, None, None, clipped_action

    def evaluate_pred(self, x, pred):
        """
        Predict distribution parameters from x (obs features)
        and returns predicted mu value of the distribution.
        Ignores the pred input parameter.

        Parameters
        ----------
        x : torch.tensor
            Feature maps extracted from environment observations.
        pred : torch.tensor
            Prediction to evaluate.

        Returns
        -------
        logp : torch.tensor
            Log probability of `pred` according to the predicted
             distribution.
        entropy_dist : torch.tensor
            Entropy of the predicted distribution.
        dist : torch.Distribution
            Action probability distribution.
        """
        pred = torch.tanh(self.action_output(x))
        pred = torch.clamp(pred, min=-1, max=1)

        return None, None, pred

class DeterministicMB(nn.Module):
    """Deterministic ensemble output layer 
    
        Parameters
        ----------
        num_inputs: int
            Size of input feature maps.
        num_outputs: int
            Output size of the gaussian layer.
        ensemble_size: int
            Ensemble size in the output layer.
    """
    def __init__(self, num_inputs: int, num_outputs: int)-> None:
        super(DeterministicMB, self).__init__()

        self.num_outputs = num_outputs
        self.output = nn.Linear(in_features=num_inputs, out_features=num_outputs)
        
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """Forward pass"""
        mean = self.output(x)
        return mean
