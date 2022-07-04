import gym
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim

import pytorchrl as prl
from pytorchrl.agent.algorithms.base import Algorithm
from pytorchrl.agent.algorithms.utils import get_gradients, set_gradients


class MPC_CEM(Algorithm):
    """
    Model-Based MPC Cross-Entropy Method (CEM) class.
    Trains a model of the environment and uses CEM to select actions.

    Parameters
    ----------
    lr: float
        Dynamics model learning rate.
    envs : VecEnv
        Vector of environments instance.
    actor : Actor
        actor class instance.
    device : torch.device
        CPU or specific GPU where class computations will take place.
    mb_epochs : int
        Training epochs for the dynamics model.
    start_steps: int
        Number of steps collected with initial random policy.
    update_every : int
         Amount of data collected in between dynamics model updates.
    action_noise :
        Exploration noise.
    mini_batch_size : int
        Size of actor update batches.
    ub : float
        Actions upper bound.
    lb : float
        Actions lower bound.
    k_best : int
        Number of best action proposals per iteration.
    epsilon : float
        Threshold to stop the training iteration earlier if the action variance is very low.
    update_alpha :
        Action distribution mean soft update parameter.
    iter_update_steps :
        Number of optimizing action sampling iterations.
    max_grad_norm : float
        Gradient clipping parameter.
    test_every : int
        Regularity of test evaluations.
    num_test_episodes : int
        Number of episodes to complete in each test phase.
    """

    def __init__(self,
                 lr,
                 envs,
                 actor,
                 device,
                 start_steps,
                 update_every,
                 mb_epochs,
                 action_noise,
                 mini_batch_size,
                 ub=1,
                 lb=-1,
                 k_best=5,
                 epsilon=0.001,
                 update_alpha=0.0,
                 max_grad_norm=0.5,
                 iter_update_steps=3,
                 test_every=10,
                 num_test_episodes=3):

        # ---- General algo attributes ----------------------------------------

        # Number of steps collected with initial random policy
        self._start_steps = int(start_steps)

        # Times data in the buffer is re-used before data collection proceeds
        self._num_epochs = int(mb_epochs)

        # Tracks the number of times data is reused
        self.mb_train_epochs = 0

        # Number of data samples collected between network update stages
        self._update_every = int(update_every)

        # Number mini batches per epoch
        self._num_mini_batch = int(1)  # Depends on how much data is available

        # Size of update mini batches
        self._mini_batch_size = int(mini_batch_size)

        # Number of network updates between test evaluations
        self._test_every = int(test_every)

        # Number of episodes to complete when testing
        self._num_test_episodes = num_test_episodes

        # ---- CEM-specific attributes ----------------------------------------

        # Number of episodes to complete when testing
        self.iter = 0
        self.envs = envs
        self.actor = actor
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.reuse_data = False
        self.action_noise = action_noise

        assert isinstance(self.actor.dynamics_model.action_space, gym.spaces.Box),\
            "CEM requires a continuous action space!"

        self.lb = lb
        self.ub = ub
        self.k_best = k_best
        self.epsilon = epsilon
        self.update_alpha = update_alpha
        self.iter_update_steps = iter_update_steps

        # ----- Optimizers ----------------------------------------------------

        self.dynamics_optimizer = optim.Adam(self.actor.dynamics_model.parameters(), lr=lr)
        self.loss_func = torch.nn.MSELoss()

    @classmethod
    def create_factory(cls,
                       lr,
                       start_steps,
                       update_every,
                       mb_epochs,
                       action_noise,
                       mini_batch_size,
                       ub=1,
                       lb=-1,
                       k_best=5,
                       epsilon=0.001,
                       update_alpha=0.0,
                       max_grad_norm=0.5,
                       iter_update_steps=3,
                       test_every=10,
                       num_test_episodes=3):
        """
        Returns a function to create a new Model-Based MPC instance.

        lr: float
            Dynamics model learning rate.
        mb_epochs : int
            Training epochs for the dynamics model.
        start_steps: int
            Number of steps collected with initial random policy.
        update_every : int
             Amount of data collected in between dynamics model updates.
        action_noise :
            Exploration noise.
        mini_batch_size : int
            Size of actor update batches.
        ub : float
            Actions upper bound.
        lb : float
            Actions lower bound.
        k_best : int
            Number of best action proposals per iteration.
        epsilon : float
            Threshold to stop the training iteration earlier if the action variance is very low.
        update_alpha :
            Action distribution mean soft update parameter.
        iter_update_steps :
            Number of optimizing action sampling iterations.
        max_grad_norm : float
            Gradient clipping parameter.
        test_every : int
            Regularity of test evaluations.
        num_test_episodes : int
            Number of episodes to complete in each test phase.

        Returns
        -------
        create_algo_instance : func
            Function that creates a new MPC_CEM class instance.
        algo_name : str
            Name of the algorithm.
        """

        def create_algo_instance(device, actor, envs):
            return cls(lr=lr,
                       envs=envs,
                       actor=actor,
                       device=device,
                       mb_epochs=update_every,
                       start_steps=start_steps,
                       update_every=update_every,
                       action_noise=action_noise,
                       mini_batch_size=mini_batch_size,
                       test_every=test_every,

                       k_best=k_best,
                       update_alpha=update_alpha,
                       iter_update_steps=iter_update_steps,

                       lb=lb,
                       ub=ub,
                       epsilon=epsilon,
                       max_grad_norm=max_grad_norm,
                       )

        return create_algo_instance, prl.MPC_RS

    @property
    def gamma(self):
        """Returns discount factor gamma."""
        return None

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

    def select_k_best(self, rewards, action_hist):
        """Selects k action trajectories that led to the highest reward.

        Parameters
        ----------
        rewards: np.array
            Rewards per rollout
        action_history: np.array
            Action history for all rollouts

        Returns
        -------
        k_best_rewards: np.array
            K-rewards of the action trajectories that the highest reward value
        elite_actions: np.array
            Best action histories
        """
        assert rewards.shape == (self.actor.n_planner, 1)
        idxs = np.argsort(rewards, axis=0)

        elite_actions = action_hist[idxs][-self.k_best:, :].squeeze(1)  # sorted (elite, horizon x action_dims)
        k_best_rewards = rewards[idxs][-self.k_best:, :].squeeze(-1)

        assert k_best_rewards.shape == (self.k_best, 1)
        assert elite_actions.shape == (self.k_best, self.actor.horizon * self.actor.action_dims)
        return k_best_rewards, elite_actions

    def update_gaussians(self, old_mu, old_var, best_actions):
        """Updates the mu and var value for the gaussian action sampling method.

        Parameters
        ----------
        old_mu: np.array
            Old mean value
        old_var: np.array
            Old variance value
        best_actions: np.array
            Action history that led to the highest reward

        Returns
        -------
        mu: np.array
            Updated mean values
        var: np.array
            Updated variance values

        """
        assert best_actions.shape == (self.k_best, self.actor.horizon * self.actor.action_dims)

        new_mu = best_actions.mean(0)
        new_var = best_actions.var(0)

        # Softupdate
        mu = (self.update_alpha * old_mu + (1.0 - self.update_alpha) * new_mu)
        var = (self.update_alpha * old_var + (1.0 - self.update_alpha) * new_var)
        assert mu.shape == (self.actor.horizon * self.actor.action_dims,)
        assert var.shape == (self.actor.horizon * self.actor.action_dims,)
        return mu, var

    def acting_step(self, obs, rhs, done, deterministic=False):
        """
        Does the MPC search with CEM action planning process.

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
            Additional MPC predictions, which are not used in other algorithms.
        """

        with torch.no_grad():

            initial_state = obs.repeat(self.actor.n_planner, 1).to(self.device)
            mu = np.zeros(self.actor.horizon * self.actor.action_dims)
            var = 5 * np.ones(self.actor.horizon * self.actor.action_dims)
            X = stats.truncnorm(self.lb, self.ub, loc=np.zeros_like(mu), scale=np.ones_like(mu))
            i = 0

            while i < self.iter_update_steps and np.max(var) > self.epsilon:
                states = initial_state
                returns = np.zeros((self.actor.n_planner, 1))
                # variables
                lb_dist = mu - self.lb
                ub_dist = self.ub - mu
                constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

                actions = X.rvs(size=[self.actor.n_planner, self.actor.horizon * self.actor.action_dims]) * np.sqrt(
                    constrained_var) + mu
                actions = np.clip(actions, self.lb, self.ub)
                actions_t = torch.from_numpy(actions.reshape(
                    self.actor.n_planner, self.actor.horizon, self.actor.action_dims)).float().to(self.device)

                for t in range(self.actor.horizon):
                    with torch.no_grad():
                        states, rewards = self.actor.dynamics_model.predict(states, actions_t[:, t, :])
                    returns += rewards.cpu().numpy()

                k_best_rewards, k_best_actions = self.select_k_best(returns, actions)
                mu, var = self.update_gaussians(mu, var, k_best_actions)
                i += 1

            best_action_sequence = mu.reshape(self.actor.horizon, -1)
            best_action = np.copy(best_action_sequence[0])
            assert best_action.shape == (self.actor.action_dims,)
            action = torch.from_numpy(best_action).float().to(self.device)
            clipped_action = action

        return action.unsqueeze(0), clipped_action.unsqueeze(0), rhs, {}

    def training_step(self, batch):
        """Does the forward pass and loss calculation of the dynamics model given the training data.

        Parameters
        ----------
        batch: dict
            Training data with inputs and labels

        Returns
        -------
            torch.Tensor: Returns the training loss
        """
        train_inputs = batch["train_input"]
        train_labels = batch["train_label"]

        self.actor.train()
        prediction = self.actor.dynamics_model.model(train_inputs)
        loss = self.loss_func(prediction, train_labels)
        return loss

    def compute_gradients(self, batch, grads_to_cpu=True):
        """
        Compute loss and compute gradients but don't do optimization step,
        return gradients instead.

        Parameters
        ----------
        batch: dict
            data batch containing all required tensors to compute dynamics model losses.
        grads_to_cpu: bool
            If gradient tensor will be sent to another node, need to be in CPU.

        Returns
        -------
        grads: list of tensors
            List of actor_critic gradients.
        info: dict
            Dict containing current dynamics model iteration information.
        """
        if batch["batch_number"] == 0:

            # TODO: add reinitialization
            # reinitializes model for new training
            # if self.iter != 0 and self.mb_train_epochs == 0:
            #     self.actor.reinitialize_dynamics_model()
            #     self.actor.to(self.device)
            #     self.dynamics_optimizer = optim.Adam(self.actor.dynamics_model.parameters(), lr=self.lr)

            self.reuse_data = True
            self.mb_train_epochs += 1

        if self.mb_train_epochs == self.num_epochs:
            self.reuse_data = False
            self.mb_train_epochs = 0

        train_loss = self.training_step(batch)

        self.dynamics_optimizer.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.dynamics_model.parameters(), self.max_grad_norm)
        dyna_grads = get_gradients(self.actor.dynamics_model, grads_to_cpu=grads_to_cpu)

        info = {"train_loss": train_loss.item()}
        grads = {"dyna_grads": dyna_grads}

        # once break condition is used set reuse_data to False
        return grads, info

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
                self.actor.dynamics_model,
                gradients=gradients["dyna_grads"], device=self.device)

        self.dynamics_optimizer.step()
        self.iter += 1

    def set_weights(self, actor_weights):
        """
        Update actor with the given weights. Update also target networks.

        Parameters
        ----------
        actor_weights : dict of tensors
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
        if parameter_name == "lr":
            for param_group in self.dynamics_optimizer.param_groups:
                param_group['lr'] = new_parameter_value
