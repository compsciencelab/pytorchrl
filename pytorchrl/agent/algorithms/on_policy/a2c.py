import torch
import itertools
import torch.nn as nn
import torch.optim as optim

import pytorchrl as prl
from pytorchrl.agent.algorithms.base import Algorithm
from pytorchrl.agent.algorithms.policy_loss_addons import PolicyLossAddOn
from pytorchrl.agent.algorithms.utils import get_gradients, set_gradients


class A2C(Algorithm):
    """
    Algorithm class to execute A2C, from Mnih et al. 2016 (https://arxiv.org/pdf/1602.01783.pdf).

    Parameters
    ----------
    device : torch.device
        CPU or specific GPU where class computations will take place.
    envs : VecEnv
        Vector of environments instance.
    actor : Actor
        Actor_critic class instance.
    lr_v : float
        Value network learning rate.
    lr_pi : float
        Policy network learning rate.
    gamma : float
        Discount factor parameter.
    num_test_episodes : int
        Number of episodes to complete in each test phase.
    max_grad_norm : float
        Gradient clipping parameter.
    test_every : int
        Regularity of test evaluations in actor updates.
    num_test_episodes : int
        Number of episodes to complete in each test phase.
    policy_loss_addons : list
        List of PolicyLossAddOn components adding loss terms to the algorithm policy loss.
    """

    def __init__(self,
                 device,
                 envs,
                 actor,
                 lr_v=1e-4,
                 lr_pi=1e-4,
                 gamma=0.99,
                 test_every=5000,
                 max_grad_norm=0.5,
                 num_test_episodes=5,
                 policy_loss_addons=[]):

        # ---- General algo attributes ----------------------------------------

        # Discount factor
        self._gamma = gamma

        # Number of steps collected with initial random policy
        self._start_steps = int(0)  # Default to 0 for On-policy algos

        # Times data in the buffer is re-used before data collection proceeds
        self._num_epochs = int(1)

        # Number of data samples collected between network update stages
        self._update_every = None  # Depends on storage capacity

        # Number mini batches per epoch
        self._num_mini_batch = int(1)

        # Size of update mini batches
        self._mini_batch_size = None  # Depends on storage capacity

        # Number of network updates between test evaluations
        self._test_every = int(test_every)

        # Number of episodes to complete when testing
        self._num_test_episodes = int(num_test_episodes)

        # ---- A2C-specific attributes ----------------------------------------

        self.iter = 0
        self.envs = envs
        self.actor = actor
        self.device = device
        self.max_grad_norm = max_grad_norm

        assert hasattr(self.actor, "value_net1"), "A2C requires value critic (num_critics=1)"

        # ----- Optimizer -----------------------------------------------------

        self.pi_optimizer = optim.Adam(self.actor.policy_net.parameters(), lr=lr_pi)
        self.v_optimizer = optim.Adam(self.actor.value_net1.parameters(), lr=lr_v)

        # ----- Policy Loss Addons --------------------------------------------

        # Sanity check, policy_loss_addons is a PolicyLossAddOn instance
        # or a list of PolicyLossAddOn instances
        assert isinstance(policy_loss_addons, (PolicyLossAddOn, list)),\
            "A2C policy_loss_addons parameter should be a  PolicyLossAddOn instance " \
            "or a list of PolicyLossAddOn instances"
        if isinstance(policy_loss_addons, list):
            for addon in policy_loss_addons:
                assert isinstance(addon, PolicyLossAddOn), \
                    "A2C policy_loss_addons parameter should be a  PolicyLossAddOn" \
                    " instance or a list of PolicyLossAddOn instances"
        else:
            policy_loss_addons = [policy_loss_addons]

        self.policy_loss_addons = policy_loss_addons
        for addon in self.policy_loss_addons:
            addon.setup(self.device)

    @classmethod
    def create_factory(cls,
                       lr_v=1e-4,
                       lr_pi=1e-4,
                       gamma=0.99,
                       test_every=5000,
                       max_grad_norm=0.5,
                       num_test_episodes=5,
                       policy_loss_addons=[]):
        """
        Returns a function to create new A2C instances.

        Parameters
        ----------
        lr_v : float
            Value network learning rate.
        lr_pi : float
            Policy network learning rate.
        gamma : float
            Discount factor parameter.
        num_test_episodes : int
            Number of episodes to complete in each test phase.
        max_grad_norm : float
            Gradient clipping parameter.
        test_every : int
            Regularity of test evaluations in actor updates.
        num_test_episodes : int
            Number of episodes to complete in each test phase.
        policy_loss_addons : list
            List of PolicyLossAddOn components adding loss terms to the algorithm policy loss.

        Returns
        -------
        create_algo_instance : func
            Function that creates a new A2C class instance.
        algo_name : str
            Name of the algorithm.
        """

        def create_algo_instance(device, actor, envs):
            return cls(lr_pi=lr_pi,
                       lr_v=lr_v,
                       envs=envs,
                       actor=actor,
                       gamma=gamma,
                       device=device,
                       test_every=test_every,
                       max_grad_norm=max_grad_norm,
                       num_test_episodes=num_test_episodes,
                       policy_loss_addons=policy_loss_addons)

        return create_algo_instance, prl.A2C

    @property
    def gamma(self):
        """Returns discount factor gamma."""
        return self._gamma

    @property
    def start_steps(self):
        """Returns the number of steps to collect with initial random policy."""
        return self._start_steps

    @property
    def num_epochs(self):
        """
        Returns the number of times the whole buffer is re-used before data
        collection proceeds.
        """
        return self._num_epochs

    @property
    def update_every(self):
        """
        Returns the number of data samples collected between
        network update stages.
        """
        return self._update_every

    @property
    def num_mini_batch(self):
        """
        Returns the number of times the whole buffer is re-used before data
        collection proceeds.
        """
        return self._num_mini_batch

    @property
    def mini_batch_size(self):
        """
        Returns the number of mini batches per epoch.
        """
        return self._mini_batch_size

    @property
    def test_every(self):
        """Number of network updates between test evaluations."""
        return self._test_every

    @property
    def num_test_episodes(self):
        """
        Returns the number of episodes to complete when testing.
        """
        return self._num_test_episodes

    def acting_step(self, obs, rhs, done, deterministic=False):
        """
        A2C acting function.

        Parameters
        ----------
        obs: torch.tensor
            Current world observation
        rhs: torch.tensor
            RNN recurrent hidden state (if policy is not a RNN, rhs will contain zeroes).
        done: torch.tensor
            1.0 if current obs is the last one in the episode, else 0.0.
        deterministic: bool
            Whether to randomly sample action from predicted distribution or take the mode.

        Returns
        -------
        action: torch.tensor
            Predicted next action.
        clipped_action: torch.tensor
            Predicted next action (clipped to be within action space).
        rhs: torch.tensor
            Policy recurrent hidden state (if policy is not a RNN, rhs will contain zeroes).
        other: dict
            Additional A2C predictions, value score and action log probability.
        """

        with torch.no_grad():

            (action, clipped_action, logp_action, rhs,
             entropy_dist, dist) = self.actor.get_action(
                obs, rhs, done, deterministic)

            value_dict = self.actor.get_value(obs, rhs, done)
            value = value_dict.get("value_net1")
            rhs = value_dict.get("rhs")

            other = {prl.VAL: value, prl.LOGP: logp_action}

        return action, clipped_action, rhs, other

    def compute_loss(self, data):
        """
        Calculate A2C loss

        Parameters
        ----------
        data: dict
            Data batch dict containing all required tensors to compute A2C loss.

        Returns
        -------
        loss : torch.tensor
            A2C loss.
        """

        o, rhs, a, old_v = data[prl.OBS], data[prl.RHS], data[prl.ACT], data[prl.VAL]
        r, d, old_logp, adv = data[prl.RET], data[prl.DONE], data[prl.LOGP], data[prl.ADV]

        # Policy loss
        logp, dist_entropy, dist = self.actor.evaluate_actions(o, rhs, d, a)
        pi_loss = - (logp * adv).mean()

        # Extend policy loss with addons
        for addon in self.policy_loss_addons:
            pi_loss += addon.compute_loss_term(self.actor, dist, data)

        # Value loss
        new_v = self.actor.get_value(o, rhs, d).get("value_net1")
        value_loss = (r - new_v).pow(2).mean()

        return pi_loss, value_loss

    def compute_gradients(self, batch, grads_to_cpu=True):
        """
        Compute loss and compute gradients but don't do optimization step,
        return gradients instead.

        Parameters
        ----------
        data: dict
            data batch containing all required tensors to compute A2C loss.
        grads_to_cpu: bool
            If gradient tensor will be sent to another node, need to be in CPU.

        Returns
        -------
        grads: list of tensors
            List of actor gradients.
        info: dict
            Dict containing current A2C iteration information.
        """

        # Compute A2C losses
        action_loss, value_loss = self.compute_loss(batch)

        # Compute policy gradients
        self.pi_optimizer.zero_grad()
        action_loss.backward(retain_graph=True)

        for p in self.actor.policy_net.parameters():
            p.requires_grad = False

        # Compute value gradients
        self.v_optimizer.zero_grad()
        value_loss.backward()

        for p in self.actor.policy_net.parameters():
            p.requires_grad = True

        # Clip gradients to max value
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

        pi_grads = get_gradients(self.actor.policy_net, grads_to_cpu=grads_to_cpu)
        v_grads = get_gradients(self.actor.value_net1, grads_to_cpu=grads_to_cpu)
        grads = {"pi_grads": pi_grads, "v_grads": v_grads}

        info = {
            "value_loss": value_loss.item(),
            "action_loss": action_loss.item(),
        }

        return grads, info

    def apply_gradients(self, gradients=None):
        """
        Take an optimization step, previously setting new gradients if provided.

        Parameters
        ----------
        gradients: list of tensors
            List of actor gradients.
        """
        if gradients is not None:
            set_gradients(
                self.actor.policy_net,
                gradients=gradients["pi_grads"], device=self.device)
            set_gradients(
                self.actor.value_net1,
                gradients=gradients["v_grads"], device=self.device)

        self.pi_optimizer.step()
        self.v_optimizer.step()
        self.iter += 1

    def set_weights(self, actor_weights):
        """
        Update actor with the given weights.

        Parameters
        ----------
        actor_weights: dict of tensors
            Dict containing actor weights to be set.
        """
        self.actor.load_state_dict(actor_weights)
        self.iter += 1

    def update_algorithm_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of the algorithm, change its value
        to `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Worker.algo attribute name
        new_parameter_value : int or float
            New value for `parameter_name`.
        """
        if hasattr(self, parameter_name):
            setattr(self, parameter_name, new_parameter_value)
        if parameter_name == "lr_v":
            for param_group in self.v_optimizer.param_groups:
                param_group['lr'] = new_parameter_value
        elif parameter_name == "lr_pi":
            for param_group in self.pi_optimizer.param_groups:
                param_group['lr'] = new_parameter_value
