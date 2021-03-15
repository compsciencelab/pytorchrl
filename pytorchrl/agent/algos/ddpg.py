import itertools
import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim

from pytorchrl.agent.algos.base import Algo


class DDPG(Algo):
    """
    Deep Deterministic Policy Gradient algorithm class.

    Algorithm class to execute DDPG, from Timothy P. Lillicrap et al.
    CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING
    (https://arxiv.org/pdf/1509.02971.pdf). Algorithms are modules generally
    required by multiple workers, so DDPG.algo_factory(...) returns a function
    that can be passed on to workers to instantiate their own DDPG module.

    Parameters
    ----------
    device : torch.device
        CPU or specific GPU where class computations will take place.
    actor_critic : ActorCritic
        Actor_critic class instance.
    lr_pi : float
        Policy optimizer learning rate.
    lr_q : float
        Q-nets optimizer learning rate.
    gamma : float
        Discount factor parameter.
    polyak: float
        SAC polyak averaging parameter.
    num_updates: int
        Num consecutive actor_critic updates before data collection continues.
    update_every: int
        Regularity of actor_critic updates in number environment steps.
    start_steps: int
        Num of initial random environment steps before learning starts.
    mini_batch_size: int
        Size of actor_critic update batches.
    target_update_interval: float
        regularity of target nets updates with respect to actor_critic Adam updates.
    num_test_episodes : int
        Number of episodes to complete in each test phase.
    test_every : int
        Regularity of test evaluations in actor_critic updates.

    Examples
    --------
    >>> create_algo = DDPG.create_factory(
            lr_q=1e-4, lr_pi=1e-4, gamma=0.99, polyak=0.995,
            num_updates=50, update_every=50, test_every=5000, start_steps=20000,
            mini_batch_size=64, num_test_episodes=0, target_update_interval=1)
    """
    
    # TODO: 1. Soft update! 

    def __init__(self,
                 device,
                 actor_critic,
                 lr_q=1e-4,
                 lr_pi=1e-4,
                 gamma=0.99,
                 polyak=0.995,
                 num_updates=1,
                 update_every=50,
                 test_every=1000,
                 start_steps=20000,
                 mini_batch_size=64,
                 num_test_episodes=5,
                 target_update_interval=1):

        # ---- General algo attributes ----------------------------------------

        # Number of steps collected with initial random policy
        self.start_steps = start_steps

        # Times data in the buffer is re-used before data collection proceeds
        self.num_epochs = 1 # Default to 1 for off-policy algorithms

        # Number of data samples collected between network update stages
        self.update_every = update_every

        # Number mini batches per epoch
        self.num_mini_batch = num_updates

        # Size of update mini batches
        self.mini_batch_size = mini_batch_size

        # Number of network updates between test evaluations
        self.test_every = test_every

        # Number of episodes to complete when testing
        self.num_test_episodes = num_test_episodes

        # ---- DDPG-specific attributes ----------------------------------------

        self.iter = 0
        self.gamma = gamma
        self.polyak = polyak
        self.device = device
        self.actor_critic = actor_critic
        self.target_update_interval = target_update_interval

        # Create target networks
        self.actor_critic_targ = deepcopy(actor_critic)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks
        q_params = itertools.chain(self.actor_critic.q1.parameters())

        # List of parameters for both Q-networks
        p_params = itertools.chain(self.actor_critic.policy_net.parameters())

        # ----- Optimizers ----------------------------------------------------
        self.pi_optimizer = optim.Adam(p_params, lr=lr_pi)
        self.q_optimizer = optim.Adam(q_params, lr=lr_q)

    @classmethod
    def create_factory(cls,
                lr_q=1e-3,
                lr_pi=1e-4,
                gamma=0.99,
                polyak=0.995,
                num_updates=50,
                test_every=5000,
                update_every=50,
                start_steps=1000,
                mini_batch_size=64,
                num_test_episodes=5,
                target_update_interval=1.0):
        """
        Returns a function to create new DDPG instances.

        Parameters
        ----------
        lr_pi : float
            Policy optimizer learning rate.
        lr_q : float
            Q-nets optimizer learning rate.
        gamma : float
            Discount factor parameter.
        polyak: float
            DDPG polyak averaging parameter.
        num_updates: int
            Num consecutive actor_critic updates before data collection continues.
        update_every: int
            Regularity of actor_critic updates in number environment steps.
        start_steps: int
            Num of initial random environment steps before learning starts.
        mini_batch_size: int
            Size of actor_critic update batches.
        target_update_interval: float
            regularity of target nets updates with respect to actor_critic Adam updates.
        num_test_episodes : int
            Number of episodes to complete in each test phase.
        test_every : int
            Regularity of test evaluations in actor_critic updates.

        Returns
        -------
        create_algo_instance : func
            creates a new DDPG class instance.
        """

        def create_algo_instance(device, actor):
            return cls(lr_q=lr_q,
                       lr_pi=lr_pi,
                       gamma=gamma,
                       device=device,
                       polyak=polyak,
                       actor_critic=actor,
                       test_every=test_every,
                       start_steps=start_steps,
                       num_updates=num_updates,
                       update_every=update_every,
                       mini_batch_size=mini_batch_size,
                       num_test_episodes=num_test_episodes,
                       target_update_interval=target_update_interval)
        return create_algo_instance

    def acting_step(self, obs, rhs, done, deterministic=False):
        """
        DDPG acting function.

        Parameters
        ----------
        obs: torch.tensor
            Current world observation
        rhs: torch.tensor
            RNN recurrent hidden state (if policy is not a RNN, rhs will contain zeroes).
        done: torch.tensor
            1.0 if current obs is the last one in the episode, else 0.0.
        deterministic: bool
            Whether to randomly sample action from predicted distribution or taking the mode.

        Returns
        -------
        action: torch.tensor
            Predicted next action.
        clipped_action: torch.tensor
            Predicted next action (clipped to be within action space).
        rhs: torch.tensor
            Policy recurrent hidden state (if policy is not a RNN, rhs will contain zeroes).
        other: dict
            Additional SAC predictions, which are not used in other algorithms.
        """

        with torch.no_grad():
            (action, clipped_action, logp_action, rhs,
             entropy_dist) = self.actor_critic.get_action(
                obs, rhs, done, deterministic=deterministic)

        return action, clipped_action, rhs, {}

    def compute_loss_q(self, data, rnn_hs, n_step=1, weights=1):
        """
        Calculate SAC Q-nets loss

        Parameters
        ----------
        data: dict
            Data batch dict containing all required tensors to compute DDPG losses.
        rnn_hs : torch.tensor
            Policy recurrent hidden state obtained with the current ActorCritic version.

        Returns
        -------
        loss_q1 : torch.tensor
            Q1-net loss.
        loss_q2 : torch.tensor
            Q2-net loss.
        loss_q : torch.tensor
            Weighted average of loss_q1 and loss_q2.
        """

        o, a, r, o2, d = data["obs"], data["act"], data["rew"], data["obs2"], data["done"]

        # Q-values for all actions
        q, _ = self.actor_critic.get_q_scores(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():

            # Target actions come from *current* policy
            a2, _, _, _, _ = self.actor_critic.get_action(o2, rnn_hs, d)

            # Target Q-values
            q_pi_targ, _ = self.actor_critic_targ.get_q_scores(o2, a2)

            backup = r + (self.gamma ** n_step) * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = 0.5 * (((q - backup) ** 2) * weights).mean()

        # errors = (torch.min(q1, q2) - backup).abs()
        errors = (q - backup).abs()

        # reset Noise
        self.actor_critic.dist.noise.reset()
        
        return loss_q, errors

    def compute_loss_pi(self, data, weights=1):
        """
        Calculate DDPG policy loss.

        Parameters
        ----------
        data: dict
            Data batch dict containing all required tensors to compute DDPG losses.

        Returns
        -------
        loss_pi : torch.tensor
            DDPG policy loss.
        logp_pi : torch.tensor
            Log probability of predicted next action.
        rnn_hs : torch.tensor
            Policy recurrent hidden state obtained with the current ActorCritic version.
        """

        o, rhs, a, r, o2, d = data["obs"], data["rhs"], data["act"], data["rew"], data["obs2"], data["done"]

        pi, _, _, rnn_rhs, _ = self.actor_critic.get_action(o, rhs, d)
        q_pi, _ = self.actor_critic.get_q_scores(o, pi)

        loss_pi = -(q_pi * weights).mean()
        return loss_pi, rnn_rhs

    def compute_gradients(self, batch, grads_to_cpu=True):
        """
        Compute loss and compute gradients but don't do optimization step,
        return gradients instead.

        Parameters
        ----------
        data: dict
            data batch containing all required tensors to compute DDPG losses.
        grads_to_cpu: bool
            If gradient tensor will be sent to another node, need to be in CPU.

        Returns
        -------
        grads: list of tensors
            List of actor_critic gradients.
        info: dict
            Dict containing current SAC iteration information.
        """

        # PER
        weights = batch.pop("weights") if "weights" in batch else 1.0

        # N-step returns
        n_step = batch.pop("n_step") if "n_step" in batch else 1.0

        # Compute policy and Q losses
        loss_pi, rhs = self.compute_loss_pi(batch, weights)
        loss_q, errors = self.compute_loss_q(batch, rhs, n_step, weights)

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q.backward()

        for p in itertools.chain(self.actor_critic.q1.parameters()):
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi.backward()

        for p in itertools.chain(self.actor_critic.q1.parameters()):
            p.requires_grad = True

        grads = []
        for p in self.actor_critic.parameters():
            if grads_to_cpu:
                if p.grad is not None:
                    grads.append(p.grad.data.cpu().numpy())
                else:
                    grads.append(None)
            else:
                if p.grad is not None:
                    grads.append(p.grad)

        info = {
                "algo/loss_pi": loss_pi.detach().item(),
                "algo/loss_q1": loss_q.detach().item(),
                "algo/errors": errors.detach().cpu().numpy()
               }

        return grads, info

    def update_target_networks(self):
        """Update actor critic target networks with polyak averaging"""
        if self.iter % self.target_update_interval == 0:
            with torch.no_grad():
                for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def apply_gradients(self, gradients=None):
        """
        Take an optimization step, previously setting new gradients if provided.
        Update also target networks.

        Parameters
        ----------
        gradients: list of tensors
            List of actor_critic gradients.
        """
        if gradients is not None:
            for g, p in zip(gradients, self.actor_critic.parameters()):
                if g is not None:
                    p.grad = torch.from_numpy(g).to(self.device)

        self.q_optimizer.step()
        self.pi_optimizer.step()

        # Update target networks by polyak averaging.
        self.iter += 1
        self.update_target_networks()

    def set_weights(self, weights):
        """
        Update actor critic with the given weights. Update also target networks.

        Parameters
        ----------
        weights: dict of tensors
            Dict containing actor_critic weights to be set.
        """
        self.actor_critic.load_state_dict(weights)

        # Update target networks by polyak averaging.
        self.iter += 1
        self.update_target_networks()

    def update_algo_parameter(self, parameter_name, new_parameter_value):
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