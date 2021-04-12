import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from pytorchrl.agent.actors.utils import init


class Categorical(nn.Module):
    """
    Categorical probability distribution.

    Parameters
    ----------
    num_inputs : int
        Size of input feature maps.
    num_outputs : int
        Number of options in output space.
    """
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, deterministic=False):
        """
        Predict distribution parameters from x (obs features) and return
        predictions (sampled and clipped), sampled log
        probability and distribution entropy.

        Parameters
        ----------
        x : torch.tensor
            Feature maps extracted from environment observations.
        deterministic : bool
            Whether to randomly sample from predicted distribution or take the mode.

        Returns
        -------
        pred: torch.tensor
            Predicted value.
        clipped_pred: torch.tensor
            Predicted value (clipped to be within [-1, 1] range).
        logp : torch.tensor
            Log probability of `pred` according to the predicted distribution.
        entropy_dist : torch.tensor
            Entropy of the predicted distribution.
        """

        # Predict distribution parameters
        x = self.linear(x)

        # Create distribution and sample
        dist = torch.distributions.Categorical(logits=x)
        self.dist = dist # ugly hack to handle sac discrete case

        if deterministic:
            pred = clipped_pred = dist.probs.argmax(dim=-1, keepdim=True)
        else:
            pred = clipped_pred = dist.sample().unsqueeze(-1)

        # Action log probability
        # logp = dist.log_prob(pred.squeeze( -1)).unsqueeze(-1)
        logp = dist.log_prob(pred.squeeze(-1)).view(pred.size(0), -1).sum(-1).unsqueeze(-1)

        # Distribution entropy
        entropy_dist = dist.entropy().mean()

        return pred, clipped_pred, logp, entropy_dist


    def evaluate_pred(self, x, pred):
        """
        Return log prob of `pred` under the distribution generated from
        x (obs features). Also return entropy of the generated distribution.

        Parameters
        ----------
        x : torch.tensor
            obs feature map obtained from a policy_net.
        pred : torch.tensor
            Prediction to evaluate.

        Returns
        -------
        logp : torch.tensor
            Log probability of `pred` according to the predicted distribution.
        entropy_dist : torch.tensor
            Entropy of the predicted distribution.
        """

        # Predict distribution parameters
        x = self.linear(x)

        # Create distribution
        dist = torch.distributions.Categorical(logits=x)

        # Evaluate log prob of under dist
        logp = dist.log_prob(pred.squeeze(-1)).unsqueeze(-1).sum(-1, keepdim=True)

        # Distribution entropy
        entropy_dist = dist.entropy().mean()

        return logp, entropy_dist