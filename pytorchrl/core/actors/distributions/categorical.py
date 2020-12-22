import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from ..neural_networks.feature_extractors.utils import init

FixedCategorical = torch.distributions.Categorical

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: FixedCategorical.log_prob(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

class Categorical(nn.Module):
    """
    Categorical action distribution.

    Parameters
    ----------
    num_inputs : int
        Size of input feature maps.
    num_outputs : int
        Number of action in action space.

    Attributes
    ----------
    linear: nn.Module
        Maps the incoming feature maps to probabilities over actions in output space.
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
        actions (sampled and clipped), sampled action log
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
        x = self.linear(x)

        # Create distribution and sample action
        action_dist = torch.distributions.Categorical(logits=x)
        self.action_dist = action_dist # ugly hack to handle sac discrete case

        if deterministic:
            action = clipped_action = action_dist.probs.argmax(dim=-1, keepdim=True)
        else:
            action = clipped_action = action_dist.sample().unsqueeze(-1)

        # Action log probability
        logp_action = action_dist.log_prob(action.squeeze( -1)).unsqueeze(-1)
        logp_action = action_dist.log_prob(action.squeeze(-1)).view(action.size(0), -1).sum(-1).unsqueeze(-1)

        # Distribution entropy
        entropy_dist = action_dist.entropy().mean()

        return action, clipped_action, logp_action, entropy_dist


    def evaluate_actions(self, x, action):
        """
        Return log prob of action under the distribution generated from
        x (obs features). Also return entropy of the generated distribution.

        Parameters
        ----------
        x : torch.tensor
            obs feature map obtained from a policy_net.
        action : torch.tensor
            Evaluated action.

        Returns
        -------
        logp_action : torch.tensor
            Log probability of `action` according to the action distribution
            predicted.
        entropy_dist : torch.tensor
            Entropy of the action distribution predicted.
        """

        # Predict distribution parameters
        x = self.linear(x)

        # Create distribution
        action_dist = torch.distributions.Categorical(logits=x)

        # Evaluate log prob of action under action_dist
        logp_action = action_dist.log_prob(action.squeeze(-1)).unsqueeze(-1).sum(-1, keepdim=True)

        # Distribution entropy
        entropy_dist = action_dist.entropy().mean()

        return logp_action, entropy_dist