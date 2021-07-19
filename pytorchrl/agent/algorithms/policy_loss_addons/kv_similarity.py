import torch
import numpy as np
from torch.distributions.kl import kl_divergence

import pytorchrl as prl
from pytorchrl.agent.algorithms.policy_loss_addons import PolicyLossAddOn


class AttractionKL(PolicyLossAddOn):

    def __init__(self,
                 behavior_factories,
                 behavior_weights,
                 loss_term_weight=1.0):
        """
        Parameters
        ----------
        behavior_generators
        behavior_weights
        """

        # Check sizes match
        assert len(behavior_factories) == len(behavior_weights)

        self.behaviors = []
        self.loss_term_weight = loss_term_weight
        self.behavior_factories = behavior_factories

        # Check all behavior weights are positive
        assert (np.array(behavior_weights) >= 0.0).all()

        # Normalize behavior_weights
        self.behavior_weights = behavior_weights
        self.behavior_weights /= np.sum(self.behavior_weights)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

    def setup(self):
        """ _ """
        # Create behavior instances
        for b in self.behavior_factories:
            self.behaviors.append(b(self.device))

    def compute_loss_term(self, actor, actor_dist, data):
        """

        Parameters
        ----------
        actor
        actor_dist
        data

        Returns
        -------

        """

        o, rhs, a, d = data[prl.OBS], data[prl.RHS], data[prl.ACT], data[prl.DONE]

        if not isinstance(actor_dist, torch.distributions.Distribution):
            # If deterministic policy, use action as mean as fix scale to 1.0
            actor_dist = torch.distributions.Normal(loc=a, scale=1.0)

        # OPTION 1
        # Is this the same as KL(pi, sum of behavior dists) ???
        kl_div = 0
        for behavior, weight in zip(self.behaviors, self.behavior_weights):

            with torch.no_grad():
                _, _, dist_b = behavior.evaluate_actions(o, rhs, d, a)

            if not isinstance(dist_b, torch.distributions.Distribution):
                # If deterministic policy, use action as mean as fix scale to 1.0
                dist_b = torch.distributions.Normal(loc=dist_b, scale=1.0)

            kl_div += (weight * kl_divergence(dist_b, actor_dist)).mean()

        print("OPTION 1 {}".format(kl_div))

        # OPTION 2 FROM DEEPMIND PAPER
        # min𝑖 (𝐷𝐾𝐿 (𝑝||𝑞𝑖) − log 𝛼𝑖)
        kl_div = []
        for behavior, weight in zip(self.behaviors, self.behavior_weights):

            with torch.no_grad():
                _, _, dist_b = behavior.evaluate_actions(o, rhs, d, a)

            if not isinstance(dist_b, torch.distributions.Distribution):
                # If deterministic policy, use action as mean as fix scale to 1.0
                dist_b = torch.distributions.Normal(loc=dist_b, scale=1.0)

            kl_div.append(kl_divergence(dist_b, actor_dist) - torch.log(weight)).mean()

        import ipdb; ipdb.set_trace()
        kl_div = torch.min(kl_div)

        print("OPTION 2 {}".format(kl_div))

        return self.loss_term_weight + kl_div


class RepulsionKL(PolicyLossAddOn):

    def __init__(self,
                 behavior_factories,
                 behavior_weights,
                 attraction_weight=1.0,
                 repulsion_weight=1.0,
                 ):
        """
        Parameters
        ----------
        behavior_factories
        behavior_weights
        attraction_weight
        repulsion_weight
        """