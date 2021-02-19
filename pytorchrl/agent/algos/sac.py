import itertools
import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim

from .base import Algo


class SAC(Algo):
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
    actor_critic : ActorCritic
        Actor_critic class instance.
    lr_pi : float
        Policy optimizer learning rate.
    lr_q : float
        Q-nets optimizer learning rate.
    lr_alpha : float
        Alpha optimizer learning rate.
    gamma : float
        Discount factor parameter.
    initial_alpha: float
        Initial entropy coefficient value (temperature).
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
    reward_scaling: float
        Reward scaling factor.
    target_update_interval: float
        regularity of target nets updates with respect to actor_critic Adam updates.
    num_test_episodes : int
        Number of episodes to complete in each test phase.
    test_every : int
        Regularity of test evaluations in actor_critic updates.

    Attributes
    ----------
    start_steps : int
        Number of steps collected with initial random policy (default to 0 for
        On-policy algos).
    num_epochs : int
        Times data in the buffer is re-used before data collection proceeds.
    update_every : int
        Number of data samples collected between network update stages (depends
        on storage capacity for On-policy algos).
    num_mini_batch : int
        Number mini batches per epoch.
    mini_batch_size : int
        Size of update mini batches.
    test_every : int
        Number of network updates between test evaluations.
    num_test_episodes : int
        Num episodes to complete in each test phase.
    device : torch.device
        CPU or specific GPU where class computation take place.
    actor_critic : ActorCritic
        ActorCritic Class containing Neural Network function approximators.
    gamma : float
        Discount factor parameter.
    iter : int
        Num actor_critic Adam updates.
    polyak : float
        SAC polyak averaging parameter.
    reward_scaling : float
        Reward scaling factor.
    target_update_interval : int
        regularity of target nets updates with respect to actor_critic Adam updates.
    log_alpha : torch.tensor
        Log entropy coefficient value.
    alpha : torch.tensor
        Entropy coefficient value.
    pi_optimizer : torch.optimizer
        Policy model optimizer.
    q_optimizer : torch.optimizer
        Q critics model optimizer.
    alpha_optimizer : torch.optimizer
        alpha parameter optimizer.

    Examples
    --------
    >>> create_algo = SAC.create_factory(
            lr_q=1e-4, lr_pi=1e-4, lr_alpha=1e-4, gamma=0.99, polyak=0.995,
            num_updates=50, update_every=50, test_every=5000, start_steps=20000,
            mini_batch_size=64, alpha=1.0, reward_scaling=1.0, num_test_episodes=0,
             target_update_interval=1)
    """

    def __init__(self,
                 device,
                 actor_critic,
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
                 reward_scaling=1.0,
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

        # ---- SAC-specific attributes ----------------------------------------

        self.iter = 0
        self.gamma = gamma
        self.polyak = polyak
        self.device = device
        self.actor_critic = actor_critic
        self.reward_scaling = reward_scaling
        self.target_update_interval = target_update_interval

        if self.actor_critic.q1 is None:
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
        self.actor_critic_targ = deepcopy(actor_critic)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks
        q_params = itertools.chain(self.actor_critic.q1.parameters(),
                                   self.actor_critic.q2.parameters())

        # List of parameters for both Q-networks
        p_params = itertools.chain(self.actor_critic.policy_net.parameters(),
                                   self.actor_critic.dist.parameters())

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
                reward_scaling=1.0,
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
        initial_alpha: float
            Initial entropy coefficient value.
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
        reward_scaling: float
            Reward scaling factor.
        target_update_interval: float
            regularity of target nets updates with respect to actor_critic Adam updates.
        num_test_episodes : int
            Number of episodes to complete in each test phase.
        test_every : int
            Regularity of test evaluations in actor_critic updates.

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
                       actor_critic=actor,
                       test_every=test_every,
                       start_steps=start_steps,
                       num_updates=num_updates,
                       update_every=update_every,
                       initial_alpha=initial_alpha,
                       reward_scaling=reward_scaling,
                       mini_batch_size=mini_batch_size,
                       num_test_episodes=num_test_episodes,
                       target_update_interval=target_update_interval)
        return create_algo_instance

    @property
    def discrete_version(self):
        """Returns True if action_space is discrete."""
        return self.actor_critic.action_space.__class__.__name__ == "Discrete"

    def acting_step(self, obs, rhs, done, deterministic=False):
        """
        SAC acting function.

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

    def compute_loss_q(self, data, rnn_hs):
        """
        Calculate SAC Q-nets loss

        Parameters
        ----------
        data: dict
            Data batch dict containing all required tensors to compute SAC losses.
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
        r *= self.reward_scaling

        if self.discrete_version:

            # Q-values for all actions
            q1, q2 = self.actor_critic.get_q_scores(o)
            q1 = q1.gather(1, a.long())
            q2 = q2.gather(1, a.long())

            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                a2, _, _, _, _ = self.actor_critic.get_action(o2, rnn_hs, d)
                p_a2 = self.actor_critic.dist.dist.probs
                z = (p_a2 == 0.0).float() * 1e-8
                logp_a2 = torch.log(p_a2 + z)

                # Target Q-values
                q1_pi_targ, q2_pi_targ = self.actor_critic_targ.get_q_scores(o2)
                q_pi_targ = (p_a2 * (torch.min(q1_pi_targ, q2_pi_targ) - self.alpha * logp_a2)).sum(dim=1, keepdim=True)

                assert r.shape == q_pi_targ.shape
                backup = r + self.gamma * (1 - d) * q_pi_targ

        else:

            # Q-values for all actions
            q1, q2 = self.actor_critic.get_q_scores(o, a)

            # Bellman backup for Q functions
            with torch.no_grad():

                # Target actions come from *current* policy
                a2, _, logp_a2, _, _ = self.actor_critic.get_action(o2, rnn_hs, d)

                # Target Q-values
                q1_pi_targ, q2_pi_targ = self.actor_critic_targ.get_q_scores(o2, a2)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

                backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = 0.5 * loss_q1 + 0.5 * loss_q2
        errors = (torch.min(q1, q2) - backup).abs()

        return loss_q1, loss_q2, loss_q, errors

    def compute_loss_pi(self, data):
        """
        Calculate SAC policy loss.

        Parameters
        ----------
        data: dict
            Data batch dict containing all required tensors to compute SAC losses.

        Returns
        -------
        loss_pi : torch.tensor
            SAC policy loss.
        logp_pi : torch.tensor
            Log probability of predicted next action.
        rnn_hs : torch.tensor
            Policy recurrent hidden state obtained with the current ActorCritic version.
        """

        o, rhs, a, r, o2, d = data["obs"], data["rhs"], data["act"], data["rew"], data["obs2"], data["done"]
        r *= self.reward_scaling

        if self.discrete_version:

            pi, _, _, rnn_hs, _ = self.actor_critic.get_action(o, rhs, d)
            p_pi = self.actor_critic.dist.dist.probs
            z = (p_pi == 0.0).float() * 1e-8
            logp_pi = torch.log(p_pi + z)
            logp_pi = torch.sum(p_pi * logp_pi, dim=1, keepdim=True)
            q1_pi, q2_pi = self.actor_critic.get_q_scores(o)
            q_pi = torch.sum(torch.min(q1_pi, q2_pi) * p_pi, dim=1, keepdim=True)

        else:

            pi, _, logp_pi, rnn_hs, _ = self.actor_critic.get_action(o, rhs, d)
            q1_pi, q2_pi = self.actor_critic.get_q_scores(o, pi)
            q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        return loss_pi, logp_pi, rnn_hs

    def compute_loss_alpha(self, log_probs):
        """
        Calculate SAC entropy loss.

        Parameters
        ----------
        log_probs: torch.tensor

        Returns
        -------
        alpha_loss: torch.tensor
            SAC entropy loss.
        """
        alpha_loss = - (self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        return alpha_loss

    def calculate_target_entropy(self):
        """Calculate SAC target entropy"""
        if self.discrete_version:
            target = - np.log(1.0 / self.actor_critic.action_space.n) * 0.98
        else:
            target = - self.actor_critic.action_space.shape[0]
        return target

    def compute_gradients(self, batch, grads_to_cpu=True):
        """
        Compute loss and compute gradients but don't do optimization step,
        return gradients instead.

        Parameters
        ----------
        data: dict
            data batch containing all required tensors to compute SAC losses.
        grads_to_cpu: bool
            If gradient tensor will be sent to another node, need to be in CPU.

        Returns
        -------
        grads: list of tensors
            List of actor_critic gradients.
        info: dict
            Dict containing current SAC iteration information.
        """

        # Compute policy and Q losses
        loss_pi, logp_pi, rhs = self.compute_loss_pi(batch)
        loss_q1, loss_q2, loss_q, errors = self.compute_loss_q(batch, rhs)

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q.backward()

        for p in itertools.chain(self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters()):
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi.backward()

        for p in itertools.chain(self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters()):
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

        # Finally, next run one gradient descent step for alpha.
        self.alpha_optimizer.zero_grad()
        loss_alpha = self.compute_loss_alpha(logp_pi)
        loss_alpha.backward()

        info = {
            "algo/loss_q1": loss_q1.detach().item(),
            "algo/loss_q2": loss_q2.detach().item(),
            "algo/loss_pi": loss_pi.detach().item(),
            "algo/loss_alpha": loss_alpha.detach().item(),
            "algo/alpha": self.alpha.detach().item(),
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
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.detach().exp()

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
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.detach().exp()

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
            for param_group in self.alpha_optimizer.param_groups:
                param_group['lr'] = new_parameter_value