import numpy as np
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

        dev = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

        import ipdb; ipdb.set_trace()

    def setup(self):
        """ _ """
        # Create behavior instances
        for b in self.behavior_factories:
            import ipdb; ipdb.set_trace()
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
            import ipdb; ipdb.set_trace()
            actor_dist = torch.distributions.Normal(loc=actor_dist, scale=1.0)

        import ipdb; ipdb.set_trace()

        kl_div = 0 * torch.as_tensor(loss)
        for behavior, weight in zip(self.behaviors, self.behavior_weights):

            with torch.no_grad():
                _, dist_b = behavior.evaluate_actions(o, rhs, d, a)

            kl_div += (weight * kl_divergence(dist_b, actor_dist)).mean()

        loss_term = kl_div * self.entropy_coef / len(self.actor_behaviors)

        return loss_term


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