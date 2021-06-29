import random
import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

import pytorchrl as prl
from pytorchrl.agent.algorithms.base import Algorithm
from pytorchrl.agent.algorithms.utils import get_gradients, set_gradients


class DDQN(Algorithm):
    """
    Deep Q Learning algorithm class.

    Algorithm class to execute DQN, from Mhin et al.
    (https://www.nature.com/articles/nature14236?wm=book_wap_0005) with
    target network.

    Parameters
    ----------
    device : torch.device
        CPU or specific GPU where class computations will take place.
    actor : ActorCritic
        actor class instance.
    lr : float
        learning rate.
    gamma : float
        Discount factor parameter.
    num_updates: int
        Num consecutive actor updates before data collection continues.
    update_every: int
        Regularity of actor updates in number environment steps.
    start_steps: int
        Num of initial random environment steps before learning starts.
    mini_batch_size: int
        Size of actor update batches.
    num_test_episodes : int
        Number of episodes to complete in each test phase.
    test_every : int
        Regularity of test evaluations in actor updates.
    initial_epsilon : float
        initial value for DQN epsilon parameter.
    epsilon_decay : float
        Exponential decay rate for epsilon parameter.
    """

    def __init__(self,
                 device,
                 actor,
                 lr=1e-4,
                 gamma=0.99,
                 polyak=0.995,
                 num_updates=1,
                 update_every=50,
                 test_every=5000,
                 start_steps=20000,
                 mini_batch_size=64,
                 num_test_episodes=5,
                 initial_epsilon=1.0,
                 epsilon_decay=0.999,
                 target_update_interval=1):

        # ---- General algo attributes ----------------------------------------

        # Discount factor
        self._gamma = gamma

        # Number of steps collected with initial random policy
        self._start_steps = int(start_steps)

        # Times data in the buffer is re-used before data collection proceeds
        self._num_epochs = 1  # Default to 1 for off-policy algorithms

        # Number of data samples collected between network update stages
        self._update_every = int(update_every)

        # Number mini batches per epoch
        self._num_mini_batch = int(num_updates)

        # Size of update mini batches
        self._mini_batch_size = int(mini_batch_size)

        # Number of network updates between test evaluations
        self._test_every = int(test_every)

        # Number of episodes to complete when testing
        self._num_test_episodes = int(num_test_episodes)

        # ---- DDQN-specific attributes ---------------------------------------

        self.iter = 0
        self.device = device
        self.polyak = polyak
        self.epsilon = initial_epsilon
        self.actor = actor
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval

        # Create target network
        self.actor_targ = deepcopy(actor)

        # Freeze target networks with respect to optimizers
        for p in self.actor_targ.parameters():
            p.requires_grad = False

        # ----- Optimizer -----------------------------------------------------

        self.q_optimizer = optim.Adam(self.actor.q1.parameters(), lr=lr)

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
                       num_test_episodes=5,
                       epsilon_decay=0.999,
                       initial_epsilon=1.0,
                       target_update_interval=1):
        """
        Returns a function to create new DDQN instances.

        Parameters
        ----------
        lr : float
            learning rate.
        gamma : float
            Discount factor parameter.
        polyak: float
            Polyak averaging parameter.
        num_updates: int
            Num consecutive actor updates before data collection continues.
        update_every: int
            Regularity of actor updates in number environment steps.
        start_steps: int
            Num of initial random environment steps before learning starts.
        mini_batch_size: int
            Size of actor update batches.
        target_update_interval: float
            regularity of target nets updates with respect to actor Adam updates.
        num_test_episodes : int
            Number of episodes to complete in each test phase.
        test_every : int
            Regularity of test evaluations in actor updates.
        initial_epsilon : float
            initial value for DQN epsilon parameter.
        epsilon_decay : float
            Exponential decay rate for epsilon parameter.

        Returns
        -------
        create_algo_instance : func
            creates a new DDQN class instance.
        """

        def create_algo_instance(device, actor):
            return cls(lr=lr,
                       gamma=gamma,
                       device=device,
                       polyak=polyak,
                       actor=actor,
                       test_every=test_every,
                       start_steps=start_steps,
                       num_updates=num_updates,
                       update_every=update_every,
                       epsilon_decay=epsilon_decay,
                       mini_batch_size=mini_batch_size,
                       initial_epsilon=initial_epsilon,
                       num_test_episodes=num_test_episodes,
                       target_update_interval=target_update_interval)
        return create_algo_instance

    def acting_step(self, obs, rhs, done, deterministic=False):
        """
        DDQN acting function.

        Parameters
        ----------
        obs: torch.tensor
            Current world observation
        rhs: dict
            RNN recurrent hidden states.
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
        rhs: batch
            Actor recurrent hidden state.
        other: dict
            Additional DDQN predictions, which are not used in other algorithms.
        """

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                q, _ = self.actor.get_q_scores(obs)
                action = clipped_action = torch.argmax(q, dim=1).unsqueeze(0)
        else:
            action = clipped_action  = torch.tensor(
                [self.actor.action_space.sample()]).unsqueeze(0)

        other = {}
        return action, clipped_action, rhs, other

    def compute_loss(self, batch, n_step=1, per_weights=1):
        """
        Calculate DDQN loss

        Parameters
        ----------
        batch: dict
            Data batch dict containing all required tensors to compute DDQN loss.

        Returns
        -------
        loss : torch.tensor
            DDQN loss.
        errors : torch.tensor
            TD errors.
        """

        o, rhs, d = batch[prl.OBS], batch[prl.RHS], batch[prl.DONE]
        a, r = batch[prl.ACT], batch[prl.REW]
        o2, rhs2, d2 = batch[prl.OBS2], batch[prl.RHS2], batch[prl.DONE2]

        # Get max predicted Q values (for next states) from target model
        q_targ_vals, _ = self.actor_targ.get_q_scores(o2, rhs2, d2)
        q_targ_next = q_targ_vals.max(dim=1)[0].unsqueeze(1)

        # Compute Q targets for current states
        q_targ = r + (self.gamma ** n_step) * (1 - d2) * q_targ_next

        # Get expected Q values from local model
        q_vals, _ = self.actor.get_q_scores(o, rhs, d)
        q_exp = q_vals.gather(1, a.long())

        # Compute loss
        loss = F.mse_loss(q_targ, q_exp)

        errors = (q_exp - q_targ).abs().detach().cpu()

        return loss, errors

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
            List of actor gradients.
        info: dict
            Dict containing current DQN iteration information.
        """

        # Recurrent burn-in
        if self.actor.is_recurrent:
            batch = self.actor.burn_in_recurrent_states(batch)

        # N-step returns
        n_step = batch["n_step"] if "n_step" in batch else 1.0

        # PER
        per_weights = batch["per_weights"] if "per_weights" in batch else 1.0

        #######################################################################

        # Compute DDQN loss and gradients
        loss, errors = self.compute_loss(batch, n_step, per_weights)
        self.q_optimizer.zero_grad()
        loss.backward()
        grads = get_gradients(self.actor.q1, grads_to_cpu=grads_to_cpu)

        info = {
            "loss_q": loss.detach().item(),
            "epsilon": self.epsilon,
        }

        if "per_weights" in batch:
            info.update({"errors": errors})

        return grads, info

    def update_target_networks(self):
        """Update actor critic target networks with polyak averaging"""
        if self.iter % self.target_update_interval == 0:
            with torch.no_grad():
                for p, p_targ in zip(self.actor.parameters(), self.actor_targ.parameters()):
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
            List of actor gradients.
        """
        if gradients is not None:
            set_gradients(
                self.actor.q1, gradients=gradients, device=self.device)
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
            Dict containing actor weights to be set.
        """
        self.actor.load_state_dict(weights)

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
