import random
import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from .base import Algo


class DDQN(Algo):
    """
    Deep Q Learning algorithm class.

    Algorithm class to execute DQN, from Mhin et al.
    (https://www.nature.com/articles/nature14236?wm=book_wap_0005).

    Parameters
    ----------
    device : torch.device
        CPU or specific GPU where class computations will take place.
    actor_critic : ActorCritic
        Actor_critic class instance.
    lr : float
        learning rate.
    gamma : float
        Discount factor parameter.
    num_updates: int
        Num consecutive actor_critic updates before data collection continues.
    update_every: int
        Regularity of actor_critic updates in number environment steps.
    start_steps: int
        Num of initial random environment steps before learning starts.
    mini_batch_size: int
        Size of actor_critic update batches.
    num_test_episodes : int
        Number of episodes to complete in each test phase.
    test_every : int
        Regularity of test evaluations in actor_critic updates.
    initial_epsilon : float
        initial value for DQN epsilon parameter.
    epsilon_decay : float
        Exponential decay rate for epsilon parameter.
    """

    def __init__(self,
                 device,
                 actor_critic,
                 lr=1e-4,
                 gamma=0.99,
                 polyak=0.995,
                 num_updates=1,
                 update_every=50,
                 test_every=5000,
                 start_steps=20000,
                 mini_batch_size=64,
                 reward_scaling=1.0,
                 num_test_episodes=5,
                 initial_epsilon=1.0,
                 epsilon_decay=0.999,
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

        # ---- DQN-specific attributes ----------------------------------------

        self.iter = 0
        self.gamma = gamma
        self.device = device
        self.polyak = polyak
        self.epsilon = initial_epsilon
        self.actor_critic = actor_critic
        self.epsilon_decay = epsilon_decay
        self.reward_scaling = reward_scaling
        self.target_update_interval = target_update_interval

        # Create target network
        self.actor_critic_targ = deepcopy(actor_critic)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_targ.parameters():
            p.requires_grad = False

        # ----- Optimizer -----------------------------------------------------

        self.q_optimizer = optim.Adam(self.actor_critic.q1.parameters(), lr=lr)

    @classmethod
    def create_factory(cls,
                       lr=1e-4,
                       gamma=0.99,
                       polyak=0.995,
                       num_updates=50,
                       update_every=50,
                       test_every=5000,
                       start_steps=20000,
                       mini_batch_size=64,
                       reward_scaling=1.0,
                       num_test_episodes=5,
                       epsilon_decay=0.999,
                       initial_epsilon=1.0,
                       target_update_interval=1):
        """
        Returns a function to create new DQN instances.

        Parameters
        ----------
        lr : float
            learning rate.
        gamma : float
            Discount factor parameter.
        polyak: float
            Polyak averaging parameter.
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
        initial_epsilon : float
            initial value for DQN epsilon parameter.
        epsilon_decay : float
            Exponential decay rate for epsilon parameter.

        Returns
        -------
        create_algo_instance : func
            creates a new DQN class instance.
        """

        def create_algo_instance(device, actor):
            return cls(lr=lr,
                       gamma=gamma,
                       device=device,
                       polyak=polyak,
                       actor_critic=actor,
                       test_every=test_every,
                       start_steps=start_steps,
                       num_updates=num_updates,
                       update_every=update_every,
                       epsilon_decay=epsilon_decay,
                       reward_scaling=reward_scaling,
                       mini_batch_size=mini_batch_size,
                       initial_epsilon=initial_epsilon,
                       num_test_episodes=num_test_episodes,
                       target_update_interval=target_update_interval)
        return create_algo_instance

    def acting_step(self, obs, rhs, done, deterministic=False):
        """
        DQN acting function.

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
            Additional DQN predictions, which are not used in other algorithms.
        """

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                q, _ = self.actor_critic.get_q_scores(obs)
                action = clipped_action = torch.argmax(q, dim=1).unsqueeze(0)
        else:
            action = clipped_action  = torch.tensor(
                [self.actor_critic.action_space.sample()]).unsqueeze(0)

        other = {}
        return action, clipped_action, rhs, other

    def compute_loss(self, data):
        """
        Calculate DQN loss

        Parameters
        ----------
        data: dict
            Data batch dict containing all required tensors to compute DQN loss.
        rnn_hs : torch.tensor
            Policy recurrent hidden state obtained with the current ActorCritic version.

        Returns
        -------
        loss : torch.tensor
            DQN loss.
        """

        o, a, r, o2, d = data["obs"], data["act"], data["rew"], data["obs2"], data["done"]
        r *= self.reward_scaling

        # Get max predicted Q values (for next states) from target model
        q_targ_vals, _ = self.actor_critic_targ.get_q_scores(o2)
        q_targ_next = q_targ_vals.max(dim=1)[0].unsqueeze(1)

        # Compute Q targets for current states
        q_targ = r + self.gamma * (1 - d) * q_targ_next

        # Get expected Q values from local model
        q_vals, _ = self.actor_critic.get_q_scores(o)
        q_exp = q_vals.gather(1, a.long())

        # Compute loss
        loss = F.mse_loss(q_targ, q_exp)

        return loss

    def compute_gradients(self, batch, grads_to_cpu=True):
        """
        Compute loss and compute gradients but don't do optimization step,
        return gradients instead.

        Parameters
        ----------
        data: dict
            data batch containing all required tensors to compute DQN loss.
        grads_to_cpu: bool
            If gradient tensor will be sent to another node, need to be in CPU.

        Returns
        -------
        grads: list of tensors
            List of actor_critic gradients.
        info: dict
            Dict containing current DQN iteration information.
        """

        # Compute DQN loss
        loss = self.compute_loss(batch)

        # Compute gradients
        self.q_optimizer.zero_grad()
        loss.backward()

        grads = []
        for p in self.actor_critic.q1.parameters():
            if grads_to_cpu:
                if p.grad is not None:
                    grads.append(p.grad.data.cpu().numpy())
                else:
                    grads.append(None)
            else:
                if p.grad is not None:
                    grads.append(p.grad)

        info = {
            "algo/loss_q": loss.detach().item(),
            "algo/epsilon": self.epsilon,
        }

        return grads, info

    def update_target_networks(self):
        """Update actor critic target networks with polyak averaging"""
        if self.iter % self.target_update_interval == 0:
            with torch.no_grad():
                for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = np.clip(self.epsilon, 0.05, 1.0)

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
            for g, p in zip(gradients, self.actor_critic.q1.parameters()):
                if g is not None:
                    p.grad = torch.from_numpy(g).to(self.device)

        self.q_optimizer.step()

        # Update target networks by polyak averaging.
        self.iter += 1
        self.update_target_networks()
        self.update_epsilon()

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
        self.update_epsilon()

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
            for param_group in self.q_optimizer.param_groups:
                param_group['lr'] = new_parameter_value
