import itertools
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

import pytorchrl as prl
from pytorchrl.agent.algorithms.base import Algorithm
from pytorchrl.agent.algorithms.utils import get_gradients, set_gradients


class MB_MPC(Algorithm):


    def __init__(self,
                 device,
                 actor,
                 lr,
                 update_every=50,
                 test_every=1000,
                 max_grad_norm=0.5,
                 start_steps=20000,
                 mini_batch_size=256,
                 num_test_episodes=5
                 ):

        # ---- General algo attributes ----------------------------------------

        # Number of steps collected with initial random policy
        self._start_steps = int(start_steps)

        # Times data in the buffer is re-used before data collection proceeds
        self._num_epochs = int(1)  # Default to 1 for off-policy algorithms


        # Size of update mini batches
        self._mini_batch_size = int(mini_batch_size)

        # Number of network updates between test evaluations
        self._test_every = int(test_every)

        # Number of episodes to complete when testing
        self._num_test_episodes = int(num_test_episodes)

        # ---- DDPG-specific attributes ----------------------------------------

        self.iter = 0
        self.device = device
        self.actor = actor
        self.max_grad_norm = max_grad_norm
        self.update_every = update_every

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_targ.parameters():
            p.requires_grad = False


        # List of parameters for both Q-networks
        dynamics_params = itertools.chain(self.actor.dynamics_model.parameters())

        # ----- Optimizers ----------------------------------------------------

        self.dynamics_optimizer = optim.Adam(dynamics_params, lr=lr)

    @classmethod
    def create_factory(cls,
                       lr,
                       test_every=5000,
                       update_every=50,
                       start_steps=1000,
                       max_grad_norm=0.5,
                       mini_batch_size=64,
                       num_test_episodes=5
                       ):


        def create_algo_instance(device, actor):
            return cls(actor=actor,
                       lr=lr,
                       device=device,
                       test_every=test_every,
                       start_steps=start_steps,
                       update_every=update_every,
                       max_grad_norm=max_grad_norm,
                       mini_batch_size=mini_batch_size,
                       num_test_episodes=num_test_episodes)

        return create_algo_instance

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


        with torch.no_grad():
            action = self.actor.get_action(obs, deterministic=deterministic)

        return action, None, None, {}

    def compute_dynamics_loss(self, data, n_step=1, per_weights=1):
        """
         Calculate DDPG Q-nets loss

        Parameters
        ----------
        data: dict
            Data batch dict containing all required tensors to compute TD3 losses.
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

        o, rhs, d = data[prl.OBS], data[prl.RHS], data[prl.DONE]
        a, r = data[prl.ACT], data[prl.REW]
        o2, rhs2, d2 = data[prl.OBS2], data[prl.RHS2], data[prl.DONE2]

        # Q-values for all actions
        q = self.actor.get_q_scores(o, rhs, d, a).get("q1")

        # Bellman backup for Q functions
        with torch.no_grad():

            # Target actions come from *current* policy
            a2, _, _, _, _, dist = self.actor.get_action(o2, rhs2, d2)

            # Target Q-values
            q_targ = self.actor_targ.get_q_scores(o2, rhs2, d2, a2).get("q1")

            backup = r + (self.gamma ** n_step) * (1 - d2) * q_targ

        # MSE loss against Bellman backup
        loss_q = 0.5 * (((q - backup) ** 2) * per_weights).mean()

        # errors = (torch.min(q1, q2) - backup).abs()
        errors = (q - backup).abs()

        # reset Noise
        self.actor.policy_net.dist.noise.reset()

        return loss_q, errors



    def compute_gradients(self, batch, grads_to_cpu=True):
        """
        Compute loss and compute gradients but don't do optimization step,
        return gradients instead.

        Parameters
        ----------
        batch: dict
            data batch containing all required tensors to compute DDPG losses.
        grads_to_cpu: bool
            If gradient tensor will be sent to another node, need to be in CPU.

        Returns
        -------
        grads: list of tensors
            List of actor_critic gradients.
        info: dict
            Dict containing current DDPG iteration information.
        """

        # Recurrent burn-in
        if self.actor.is_recurrent:
            batch = self.actor.burn_in_recurrent_states(batch)

        # PER
        per_weights = batch.pop("per_weights") if "per_weights" in batch else 1.0

        # N-step returns
        n_step = batch.pop("n_step") if "n_step" in batch else 1.0

        # First run one gradient descent step for Q1 and Q2
        loss_q, errors = self.compute_loss_q(batch, n_step, per_weights)
        self.q_optimizer.zero_grad()
        loss_q.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.q1.parameters(), self.max_grad_norm)
        q_grads = get_gradients(self.actor.q1, grads_to_cpu=grads_to_cpu)

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in itertools.chain(self.actor.q1.parameters()):
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        loss_pi = self.compute_loss_pi(batch, per_weights)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        nn.utils.clip_grad_norm_(self.actor.policy_net.parameters(), self.max_grad_norm)
        pi_grads = get_gradients(self.actor.policy_net, grads_to_cpu=grads_to_cpu)

        for p in itertools.chain(self.actor.q1.parameters()):
            p.requires_grad = True

        info = {
            "loss_pi": loss_pi.detach().item(),
            "loss_q1": loss_q.detach().item(),
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
                self.actor.q1,
                gradients=gradients["q_grads"], device=self.device)

        self.q_optimizer.step()
        self.pi_optimizer.step()

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
