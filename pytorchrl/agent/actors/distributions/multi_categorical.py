import numpy as np
import torch
import torch.nn as nn
from pytorchrl.agent.actors.utils import init


class MultiCategorical(nn.Module):
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
        super(MultiCategorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=np.sqrt(0.01))

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
        dist : torch.Distribution
            Action probability distribution.
        """

        # TODO: make sure shape of x is (bs, seq_size, num_features)

        # Get x dims
        bs, seq_size, num_features = x.shape  # TODO. review

        # TODO: unravel
        x = x.view(bs * seq_size, num_features)

        # Predict distribution parameters
        x = self.linear(x)  # TODO, shape (bs, seq_size, vocabulary_size)

        dist = torch.distributions.Categorical(logits=x)

        if deterministic:
            pred = dist.probs.argmax(dim=-1, keepdim=True)
        else:
            pred = dist.sample().unsqueeze(-1)

        # Action log probability
        logp = dist.log_prob(pred.squeeze(-1)).view(pred.size(0), -1).sum(-1).unsqueeze(-1)

        # TODO: ravel
        pred = clipped_pred = pred.view(bs, seq_size, -1)  # TODO. review
        logp = logp.view(bs, seq_size, -1).sum(1)  # TODO. review

        # Distribution entropy
        entropy_dist = dist.entropy().mean()   # TODO. Is it ok?

        # TODO. Is dist ok?

        return pred, clipped_pred, logp, entropy_dist, dist

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
        dist : torch.Distribution
            Action probability distribution.
        """
        # Get x dims
        bs, seq_size, num_features = x.shape  # TODO. review

        # TODO: unravel
        x = x.view(bs * seq_size, num_features)
        reshaped_pred = pred.view(bs * seq_size, -1)

        # Predict distribution parameters
        x = self.linear(x)

        # Create distribution
        dist = torch.distributions.Categorical(logits=x)

        # Evaluate log prob of under dist
        logp = dist.log_prob(reshaped_pred).view(pred.size(0), -1).sum(-1).unsqueeze(-1)

        # Distribution entropy
        entropy_dist = dist.entropy().mean()

        return logp, entropy_dist, dist
