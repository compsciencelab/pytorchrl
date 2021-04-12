import itertools
import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim

import pytorchrl as prl
from pytorchrl.agent.algorithms.base import Algorithm
from pytorchrl.agent.algorithms.utils import get_gradients, set_gradients


class SAC(Algorithm):
    """
    Soft Actor Critic algorithm class.

    Algorithm class to execute SAC, from Haarnoja et al.
    (https://arxiv.org/abs/1812.05905). Algorithms are modules generally
    required by multiple workers, so SAC.algo_factory(...) returns a function
    that can be passed on to workers to instantiate their own SAC module.

    Parameters
    ----------
    device : torch.device
        CPU or specific GPU where class computations will take place.
    actor : Actor
        Actor_critic class instance.
    lr_pi : float
        Policy optimizer learning rate.
    lr_q : float
        Q-nets optimizer learning rate.
    lr_alpha : float
        Alpha optimizer learning rate.
    gamma : float
        Discount factor parameter.
    initial_alpha : float
        Initial entropy coefficient value (temperature).
    polyak : float
        SAC polyak averaging parameter.
    num_updates : int
        Num consecutive actor updates before data collection continues.
    update_every : int
        Regularity of actor updates in number environment steps.
    start_steps : int
        Num of initial random environment steps before learning starts.
    mini_batch_size : int
        Size of actor update batches.
    target_update_interval : float
        regularity of target nets updates with respect to actor Adam updates.
    num_test_episodes : int
        Number of episodes to complete in each test phase.
    test_every : int
        Regularity of test evaluations in actor updates.

    Examples
    --------
    >>> create_algo = SAC.create_factory(
            lr_q=1e-4, lr_pi=1e-4, lr_alpha=1e-4, gamma=0.99, polyak=0.995,
            num_updates=50, update_every=50, test_every=5000, start_steps=20000,
            mini_batch_size=64, alpha=1.0, num_test_episodes=0, target_update_interval=1)
    """

    def __init__(self,
                 device,
                 actor,
                 lr_q=1e-4,
                 lr_pi=1e-4,
                 lr_alpha=1e-4,
                 gamma=0.99,
                 polyak=0.995,
                 num_updates=1,
                 update_every=50,
                 test_every=1000,
                 initial_alpha=1.0,
                 start_steps=20000,
                 mini_batch_size=64,
                 num_test_episodes=5,
                 target_update_interval=1):

        # ---- General algo attributes ----------------------------------------

        # Discount factor
        self._gamma = gamma

        # Number of steps collected with initial random policy
        self._start_steps = start_steps

        # Times data in the buffer is re-used before data collection proceeds
        self._num_epochs = 1 # Default to 1 for off-policy algorithms

        # Number of data samples collected between network update stages
        self._update_every = update_every

        # Number mini batches per epoch
        self._num_mini_batch = num_updates

        # Size of update mini batches
        self._mini_batch_size = mini_batch_size

        # Number of network updates between test evaluations
        self._test_every = test_every

        # Number of episodes to complete when testing
        self._num_test_episodes = num_test_episodes

        # ---- SAC-specific attributes ----------------------------------------

        self.iter = 0
        self.polyak = polyak
        self.device = device
        self.actor = actor
        self.target_update_interval = target_update_interval

        if self.actor.q2 is None:
            raise ValueError("SAC requires double q critic")

        self.log_alpha = torch.tensor(
            data=[np.log(initial_alpha)], dtype=torch.float32,
            requires_grad=True, device=device)
        self.alpha = self.log_alpha.detach().exp()

        # Compute target entropy
        target_entropy = self.calculate_target_entropy()
        self.target_entropy = torch.tensor(
            data=target_entropy, dtype=torch.float32,
            requires_grad=False, device=device)

        # Create target networks
        self.actor_targ = deepcopy(actor)

        # Freeze target networks with respect to optimizers
        for p in self.actor_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks
        q_params = itertools.chain(self.actor.q1.parameters(), self.actor.q2.parameters())

        # List of parameters for policy network
        p_params = itertools.chain(self.actor.policy_net.parameters())

        # ----- Optimizers ----------------------------------------------------

        self.pi_optimizer = optim.Adam(p_params, lr=lr_pi)
        self.q_optimizer = optim.Adam(q_params, lr=lr_q)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    @classmethod
    def create_factory(cls,
                       lr_q=1e-4,
                       lr_pi=1e-4,
                       lr_alpha=1e-4,
                       gamma=0.99,
                       polyak=0.995,
                       num_updates=50,
                       test_every=5000,
                       update_every=50,
                       start_steps=1000,
                       initial_alpha=1.0,
                       mini_batch_size=64,
                       num_test_episodes=5,
                       target_update_interval=1.0):
        """
        Returns a function to create new SAC instances.

        Parameters
        ----------
        lr_pi : float
            Policy optimizer learning rate.
        lr_q : float
            Q-nets optimizer learning rate.
        lr_alpha : float
            Alpha optimizer learning rate.
        gamma : float
            Discount factor parameter.
        initial_alpha : float
            Initial entropy coefficient value.
        polyak : float
            SAC polyak averaging parameter.
        num_updates : int
            Num consecutive actor updates before data collection continues.
        update_every : int
            Regularity of actor updates in number environment steps.
        start_steps : int
            Num of initial random environment steps before learning starts.
        mini_batch_size : int
            Size of actor update batches.
        target_update_interval : float
            regularity of target nets updates with respect to actor Adam updates.
        num_test_episodes : int
            Number of episodes to complete in each test phase.
        test_every : int
            Regularity of test evaluations in actor updates.

        Returns
        -------
        create_algo_instance : func
            creates a new SAC class instance.
        """

        def create_algo_instance(device, actor):
            return cls(lr_q=lr_q,
                       lr_pi=lr_pi,
                       lr_alpha=lr_alpha,
                       gamma=gamma,
                       device=device,
                       polyak=polyak,
                       actor=actor,
                       test_every=test_every,
                       start_steps=start_steps,
                       num_updates=num_updates,
                       update_every=update_every,
                       initial_alpha=initial_alpha,
                       mini_batch_size=mini_batch_size,
                       num_test_episodes=num_test_episodes,
                       target_update_interval=target_update_interval)
        return create_algo_instance

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

    @property
    def discrete_version(self):
        """Returns True if action_space is discrete."""
        return self.actor.action_space.__class__.__name__ == "Discrete"

    def acting_step(self, obs, rhs, done, deterministic=False):
        """
        SAC acting function.

        Parameters
        ----------
        obs : torch.tensor
            Current world observation
        rhs : torch.tensor
            RNN recurrent hidden state (if policy is not a RNN, rhs will contain zeroes).
        done : torch.tensor
            1.0 if current obs is the last one in the episode, else 0.0.
        deterministic : bool
            Whether to randomly sample action from predicted distribution or taking the mode.

        Returns
        -------
        action : torch.tensor
            Predicted next action.
        clipped_action : torch.tensor
            Predicted next action (clipped to be within action space).
        rhs : torch.tensor
            Policy recurrent hidden state (if policy is not a RNN, rhs will contain zeroes).
        other : dict
            Additional SAC predictions, which are not used in other algorithms.
        """

        with torch.no_grad():
            (action, clipped_action, logp_action, rhs,
             entropy_dist) = self.actor.get_action(
                obs, rhs, done, deterministic=deterministic)

        return action, clipped_action, rhs, {}

    def compute_loss_q(self, batch, n_step=1, per_weights=1):
        """
        Calculate SAC Q-nets loss

        Parameters
        ----------
        batch : dict
            Data batch dict containing all required tensors to compute SAC losses.
        n_step : int or float
            Number of future steps used to computed the truncated n-step return value.
        per_weights :
            Prioritized Experience Replay (PER) important sampling weights or 1.0.

        Returns
        -------
        loss_q1 : torch.tensor
            Q1-net loss.
        loss_q2 : torch.tensor
            Q2-net loss.
        loss_q : torch.tensor
            Weighted average of loss_q1 and loss_q2.
        errors : torch.tensor
            TD errors.
        """

        o, rhs, d = batch[prl.OBS], batch[prl.RHS], batch[prl.DONE]
        a, r = batch[prl.ACT], batch[prl.REW]
        o2, rhs2, d2 = batch[prl.OBS2], batch[prl.RHS2], batch[prl.DONE2]

        if self.discrete_version:

            # Q-values for all actions
            q1, q2, _ = self.actor.get_q_scores(o, rhs, d)
            q1 = q1.gather(1, a.long())
            q2 = q2.gather(1, a.long())

            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                a2, _, _, _, _ = self.actor.get_action(o2, rhs2, d2)
                p_a2 = self.actor.dist.dist.probs
                z = (p_a2 == 0.0).float() * 1e-8
                logp_a2 = torch.log(p_a2 + z)

                # Target Q-values
                q1_pi_targ, q2_pi_targ, _ = self.actor_targ.get_q_scores(o2, rhs2, d2)
                q_pi_targ = (p_a2 * (torch.min(q1_pi_targ, q2_pi_targ) - self.alpha * logp_a2)).sum(dim=1, keepdim=True)

                assert r.shape == q_pi_targ.shape
                backup = r + (self.gamma ** n_step) * (1 - d2) * q_pi_targ

        else:

            # Q-values for all actions
            q1, q2, _ = self.actor.get_q_scores(o, rhs, d, a)

            # Bellman backup for Q functions
            with torch.no_grad():

                # Target actions come from *current* policy
                a2, _, logp_a2, _, _ = self.actor.get_action(o2, rhs2, d2)

                # Target Q-values
                q1_pi_targ, q2_pi_targ, _ = self.actor_targ.get_q_scores(o2, rhs2, d2, a2)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

                backup = r + (self.gamma ** n_step) * (1 - d2) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = (((q1 - backup) ** 2) * per_weights).mean()
        loss_q2 = (((q2 - backup) ** 2) * per_weights).mean()
        loss_q = 0.5 * loss_q1 + 0.5 * loss_q2

        # errors = (torch.min(q1, q2) - backup).abs().detach().cpu()
        # errors = torch.max((q1 - backup).abs(), (q2 - backup).abs()).detach().cpu()
        errors = (0.5 * (q1 - backup).abs() + 0.5 * (q2 - backup).abs()).detach().cpu()

        return loss_q1, loss_q2, loss_q, errors

    def compute_loss_pi(self, batch, per_weights=1):
        """
        Calculate SAC policy loss.

        Parameters
        ----------
        batch : dict
            Data batch dict containing all required tensors to compute SAC losses.
        per_weights :
            Prioritized Experience Replay (PER) important sampling weights or 1.0.

        Returns
        -------
        loss_pi : torch.tensor
            SAC policy loss.
        logp_pi : torch.tensor
            Log probability of predicted next action.
        """

        o, rhs, d = batch[prl.OBS], batch[prl.RHS], batch[prl.DONE]

        if self.discrete_version:

            pi, _, _, _, _ = self.actor.get_action(o, rhs, d)
            p_pi = self.actor.dist.dist.probs
            z = (p_pi == 0.0).float() * 1e-8
            logp_pi = torch.log(p_pi + z)
            logp_pi = torch.sum(p_pi * logp_pi, dim=1, keepdim=True)
            q1_pi, q2_pi, _ = self.actor.get_q_scores(o, rhs, d)
            q_pi = torch.sum(torch.min(q1_pi, q2_pi) * p_pi, dim=1, keepdim=True)

        else:

            pi, _, logp_pi, _, _ = self.actor.get_action(o, rhs, d)
            q1_pi, q2_pi, _ = self.actor.get_q_scores(o, rhs, d, pi)
            q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = ((self.alpha * logp_pi - q_pi) * per_weights).mean()

        return loss_pi, logp_pi

    def compute_loss_alpha(self, log_probs, per_weights=1):
        """
        Calculate SAC entropy loss.

        Parameters
        ----------
        log_probs : torch.tensor
            Log probability of predicted next action.
        per_weights :
            Prioritized Experience Replay (PER) important sampling weights or 1.0.

        Returns
        -------
        alpha_loss : torch.tensor
            SAC entropy loss.
        """
        alpha_loss = - ((self.log_alpha * (log_probs + self.target_entropy).detach()) * per_weights).mean()
        return alpha_loss

    def calculate_target_entropy(self):
        """Calculate SAC target entropy"""
        if self.discrete_version:
            target = - np.log(1.0 / self.actor.action_space.n) * 0.98
        else:
            target_old = - self.actor.action_space.shape[0]
            target = - np.prod(self.actor.action_space.shape)
            assert target_old == target
        return target

    def compute_gradients(self, batch, grads_to_cpu=True):
        """
        Compute loss and compute gradients but don't do optimization step,
        return gradients instead.

        Parameters
        ----------
        batch : dict
            data batch containing all required tensors to compute SAC losses.
        grads_to_cpu : bool
            If gradient tensor will be sent to another node, need to be in CPU.

        Returns
        -------
        grads : list of tensors
            List of actor gradients.
        info : dict
            Dict containing current SAC iteration information.
        """

        # Recurrent burn-in
        if self.actor.is_recurrent:
            batch = self.actor.burn_in_recurrent_states(batch)

        # N-step returns
        n_step = batch["n_step"] if "n_step" in batch else 1.0

        # PER
        per_weights = batch["per_weights"] if "per_weights" in batch else 1.0

        # Run one gradient descent step for Q1 and Q2
        loss_q1, loss_q2, loss_q, errors = self.compute_loss_q(batch, n_step, per_weights)
        self.q_optimizer.zero_grad()
        loss_q.backward(retain_graph=True)
        q_grads = get_gradients(self.actor.q1, self.actor.q2, grads_to_cpu=grads_to_cpu)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in itertools.chain(self.actor.q1.parameters(), self.actor.q2.parameters()):
            p.requires_grad = False

        # Run one gradient descent step for pi.
        loss_pi, logp_pi = self.compute_loss_pi(batch, per_weights)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        pi_grads = get_gradients(self.actor.policy_net, grads_to_cpu=grads_to_cpu)

        for p in itertools.chain(self.actor.q1.parameters(), self.actor.q2.parameters()):
            p.requires_grad = True

        # Run one gradient descent step for alpha.
        self.alpha_optimizer.zero_grad()
        loss_alpha = self.compute_loss_alpha(logp_pi, per_weights)
        loss_alpha.backward()

        info = {
            "loss_q1": loss_q1.detach().item(),
            "loss_q2": loss_q2.detach().item(),
            "loss_pi": loss_pi.detach().item(),
            "loss_alpha": loss_alpha.detach().item(),
            "alpha": self.alpha.detach().item(),
        }

        if "per_weights" in batch:
            info.update({"errors": errors})

        grads = {"q_grads": q_grads, "pi_grads": pi_grads}

        return grads, info

    def update_target_networks(self):
        """Update actor critic target networks with polyak averaging"""
        if self.iter % self.target_update_interval == 0:
            with torch.no_grad():
                for p, p_targ in zip(self.actor.parameters(), self.actor_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def apply_gradients(self, gradients=None):
        """
        Take an optimization step, previously setting new gradients if provided.
        Update also target networks.

        Parameters
        ----------
        gradients : list of tensors
            List of actor gradients.
        """
        if gradients is not None:
            set_gradients(
                self.actor.policy_net,
                gradients=gradients["pi_grads"], device=self.device)
            set_gradients(
                self.actor.q1, self.actor.q2,
                gradients=gradients["q_grads"], device=self.device)

        self.q_optimizer.step()
        self.pi_optimizer.step()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.detach().exp()

        # Update target networks by polyak averaging.
        self.iter += 1
        self.update_target_networks()

    def set_weights(self, actor_weights):
        """
        Update actor with the given weights. Update also target networks.

        Parameters
        ----------
        actor_weights : dict of tensors
            Dict containing actor weights to be set.
        """
        self.actor.load_state_dict(actor_weights)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.detach().exp()

        # Update target networks by polyak averaging.
        self.iter += 1
        self.update_target_networks()

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
        if parameter_name == "lr":
            for param_group in self.pi_optimizer.param_groups:
                param_group['lr'] = new_parameter_value
            for param_group in self.q_optimizer.param_groups:
                param_group['lr'] = new_parameter_value
            for param_group in self.alpha_optimizer.param_groups:
                param_group['lr'] = new_parameter_value
