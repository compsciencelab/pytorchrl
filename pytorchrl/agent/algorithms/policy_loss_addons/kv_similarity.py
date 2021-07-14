from pytorchrl.agent.algorithms.policy_loss_addons import PolicyLossAddOn


class KL(PolicyLossAddOn):

    def __init__(self, behavior_factories, behavior_weights):
        """
        Parameters
        ----------
        behavior_generators
        behavior_weights
        """

        assert len(behavior_factories) == len(behavior_weights)

        self.behaviors = []
        self.behavior_weights = behavior_weights
        self.behavior_factories = behavior_factories

        dev = device or "cuda" if torch.cuda.is_available() else "cpu"
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

        # If actor distribution is not a torch.Distribution
        # do what?

        kl_div = 0 * torch.as_tensor(loss)
        for behavior, weight in zip(self.behaviors, self.behavior_weights):

            with torch.no_grad():
                _, dist_b = behavior.evaluate_actions(o, rhs, d, a)

            kl_div += (weight * kl_divergence(dist_b, actor_dist)).mean()

        loss_term = kl_div * self.entropy_coef / len(self.actor_behaviors)

        return loss_term
