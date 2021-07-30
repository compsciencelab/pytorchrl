import torch
import numpy as np
from torch.distributions.kl import kl_divergence

import pytorchrl as prl
from pytorchrl.agent.algorithms.policy_loss_addons import PolicyLossAddOn


class AttractionKL(PolicyLossAddOn):

    def __init__(self,
                 behavior_factories,
                 behavior_weights,
                 loss_term_weight=1.0,
                 eps=1e-8):
        """
        Class to enforce similarity of any algorithm policy to specified list of behaviors.
        We use the same loss term as in https://arxiv.org/pdf/2105.12196.pdf.

        Parameters
        ----------
        behavior_factories : list
            List of methods creating the agent behaviors.
        behavior_weights : list
            List of floats giving relative weight to each agent behavior. All weights should be
            positive. Otherwise AssertionError will be raised.
        loss_term_weight : float
            Weight of the KL term in the algorithm policy loss.
        eps : float
            Lower bound for prob values, used to clip action probs.
        """

        # Check sizes match
        assert len(behavior_factories) == len(behavior_weights)

        self.eps = eps
        self.behaviors = []
        self.loss_term_weight = loss_term_weight
        self.behavior_factories = behavior_factories

        # Check all behavior weights are positive
        assert (np.array(behavior_weights) >= 0.0).all()

        # Normalize behavior_weights
        self.behavior_weights = behavior_weights
        self.behavior_weights /= np.sum(self.behavior_weights)

    def setup(self, device):
        """
        Setup addon module by casting behavior weights to torch tensors and
        initializing agent behaviors.
        """

        self.device = device

        # Cast behavior weights to torch tensors
        self.behavior_weights = [torch.tensor(w).to(device) for w in self.behavior_weights]

        # Create behavior instances
        for b in self.behavior_factories:
            self.behaviors.append(b(self.device))

    def compute_loss_term(self, actor, actor_dist, data):
        """
        Calculate and add KL Attraction loss term.
            1. Calculate KL between actor policy and all behaviors.
            2. Compute biased KL similarities and select minimum value.
            3. Multiply the result by the loss_term_weight.
            4. Change sign of the loss term so KL between behaviors is minimized.

        Parameters
        ----------
        actor : Actor
            Training algorithm's Actor_critic class instance.
        actor_dist : torch.distributions.Distribution
            Actor action distribution for actions in data[prl.OBS]
        data : dict
            data batch containing all required tensors to compute loss term.

        Returns
        -------
        attraction_kl_loss_term : torch.tensor
            KL loss term.
        """

        o, rhs, a, d = data[prl.OBS], data[prl.RHS], data[prl.ACT], data[prl.DONE]

        if not isinstance(actor_dist, torch.distributions.Distribution):
            # If deterministic policy, use action as mean as fix scale to 1.0
            actor_dist = torch.distributions.Normal(loc=a, scale=1.0)

        actor_dist.probs = torch.clamp(actor_dist.probs, self.eps, 1.0 - self.eps)

        kl_div = []
        for behavior, weight in zip(self.behaviors, self.behavior_weights):

            with torch.no_grad():
                _, _, dist_b = behavior.evaluate_actions(o, rhs, d, a)

            if not isinstance(dist_b, torch.distributions.Distribution):
                # If deterministic policy, use action as mean as fix scale to 1.0
                dist_b = torch.distributions.Normal(loc=dist_b, scale=1.0)

            dist_b.probs = torch.clamp(dist_b.probs, self.eps, 1.0 - self.eps)

            div = (kl_divergence(dist_b, actor_dist) - torch.log(weight))

            # div *= torch.exp(- 2 * dist_b.entropy()).detach()

            kl_div.append(div.mean())

        kl_div = min(kl_div)

        return self.loss_term_weight * kl_div


class RepulsionKL(PolicyLossAddOn):

    def __init__(self,
                 behavior_factories,
                 behavior_weights,
                 loss_term_weight=1.0,
                 eps=1e-8):
        """
        Class to enforce dissimilarity of any algorithm policy to specified list of behaviors.

        Parameters
        ----------
        behavior_factories : list
            List of methods creating the agent behaviors.
        behavior_weights : list
            List of floats giving relative weight to each agent behavior. All weights should be
            positive. Otherwise AssertionError will be raised.
        loss_term_weight : float
            Weight of the KL term in the algorithm policy loss.
        eps : float
            Lower bound for prob values, used to clip action probs.
        """

        # Check sizes match
        assert len(behavior_factories) == len(behavior_weights)

        self.eps = eps
        self.behaviors = []
        self.loss_term_weight = loss_term_weight
        self.behavior_factories = behavior_factories

        # Check all behavior weights are positive
        assert (np.array(behavior_weights) >= 0.0).all()

        # Normalize behavior_weights
        self.behavior_weights = behavior_weights
        self.behavior_weights /= np.sum(self.behavior_weights)

    def setup(self, device):
        """
        Setup addon module by casting behavior weights to torch tensors and
        initializing agent behaviors.
        """

        self.device = device

        # Cast behavior weights to torch tensors
        self.behavior_weights = [torch.tensor(w).to(device) for w in self.behavior_weights]

        # Create behavior instances
        for b in self.behavior_factories:
            self.behaviors.append(b(self.device))

    def compute_loss_term(self, actor, actor_dist, data):
        """
        Calculate and add KL Repulsion loss term.
            1. Calculate KL between actor policy and all behaviors.
            2. Compute weighted sum of KL similarities.
            3. Multiply the result by the loss_term_weight.
            4. Keep sign of the loss term so KL between behaviors is maximized.

        Parameters
        ----------
        actor : Actor
            Training algorithm's Actor_critic class instance.
        actor_dist : torch.distributions.Distribution
            Actor action distribution for actions in data[prl.OBS]
        data : dict
            data batch containing all required tensors to compute loss term.

        Returns
        -------
        attraction_kl_loss_term : torch.tensor
            KL loss term.
        """

        o, rhs, a, d = data[prl.OBS], data[prl.RHS], data[prl.ACT], data[prl.DONE]

        if not isinstance(actor_dist, torch.distributions.Distribution):
            # If deterministic policy, use action as mean as fix scale to 1.0
            actor_dist = torch.distributions.Normal(loc=a, scale=1.0)

        actor_dist.probs = torch.clamp(actor_dist.probs, self.eps, 1.0 - self.eps)

        kl_div = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        for behavior, weight in zip(self.behaviors, self.behavior_weights):

            with torch.no_grad():
                _, _, dist_b = behavior.evaluate_actions(o, rhs, d, a)

            if not isinstance(dist_b, torch.distributions.Distribution):
                # If deterministic policy, use action as mean as fix scale to 1.0
                dist_b = torch.distributions.Normal(loc=dist_b, scale=1.0)

            dist_b.probs = torch.clamp(dist_b.probs, self.eps, 1.0 - self.eps)

            div = kl_divergence(dist_b, actor_dist)

            # div *= torch.exp(- 2 * dist_b.entropy()).detach()

            kl_div += div.mean()

        return -1 * self.loss_term_weight * kl_div
