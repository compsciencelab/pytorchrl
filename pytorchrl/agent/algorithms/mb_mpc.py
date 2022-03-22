import itertools
from typing import Tuple
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

import pytorchrl as prl
from pytorchrl.agent.algorithms.base import Algorithm
from pytorchrl.agent.algorithms.utils import get_gradients, set_gradients
from pytorchrl.agent.actors.planner import MPC


class MB_MPC(Algorithm):
    """Model-Based MPC class.
    Trains a model of the environment and uses MPC to select actions. 
    User can choose between different MPC methods:
    - Random Shooting (RS)
    - Cross Entropy Method (CEM)
    - Filtering and Reward-Weighted Refinement (PDDM) 
        as introduced in the PDDM paper: https://arxiv.org/pdf/1909.11652.pdf

    Parameters
    ----------
    device : torch.device
        CPU or specific GPU where class computations will take place.
    envs : VecEnv
        Vector of environments instance.
    actor : MBActor
        MB_actor class instance.
    config: Namespace
        Training configuration defined at the beginning of training
    """
    def __init__(self,
                 actor,
                 device,
                 envs,
                 config,
                 ):

        # ---- General algo attributes ----------------------------------------
        # Number of steps collected with initial random policy
        self._start_steps = int(config.start_steps)

        # Times data in the buffer is re-used before data collection proceeds
        self._num_epochs = int(config.mb_epochs)  # Default to 1 for off-policy algorithms
        self.mb_train_epochs = 0
        # Size of update mini batches
        self._mini_batch_size = int(config.mini_batch_size)
        self._num_mini_batch = 1
        # Number of network updates between test evaluations
        self._test_every = int(config.test_every)

        # Number of episodes to complete when testing
        self._num_test_episodes = int(3)
        self.actor = actor
        self.action_noise = config.action_noise
        # ---- MB MPC-specific attributes ----------------------------------------
        if config.mpc_type == "RS":
            self.mpc = MPC.RandomShooting(action_space=self.actor.action_space,
                                          config=config,
                                          device=device)
        elif config.mpc_type == "CEM":
            self.mpc = MPC.CEM(action_space=self.actor.action_space,
                               config=config,
                               device=device)
        elif config.mpc_type == "PDDM":
            self.mpc = MPC.PDDM(action_space=self.actor.action_space,
                                config=config,
                                device=device)
        else:
            raise ValueError

        self.iter = 0
        self.device = device
        self.max_grad_norm = 0.5
        self._update_every = config.update_every
        self.reuse_data = False

        # ----- Optimizers ----------------------------------------------------
        self.lr = config.lr
        self.dynamics_optimizer = optim.Adam(self.actor.dynamics_model.parameters(), lr=self.lr)
        self.loss_func = torch.nn.MSELoss()

    @classmethod
    def create_factory(cls,
                       config,
                       ):
        """
        Returns a function to create a new Model-Based MPC instance.
        
        Parameters
        ----------
        config: Namespace
           Includes algorithm parameter and also MPC specific parameter.

        Returns
        -------
        create_algo_instance : func
            Function that creates a new DDPG class instance.
        algo_name : str
            Name of the algorithm.
        """

        def create_algo_instance(device, actor, envs):
            return cls(actor=actor,
                       device=device,
                       envs=envs,
                       config=config,)

        return create_algo_instance, prl.MPC
    
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
        """Does the MPC search.

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
            action = self.mpc.get_action(state=obs, model=self.actor, noise=False)

            clipped_action = torch.clamp(action, -1, 1)
            
        if self.actor.unscale:
            action = self.actor.unscale(action)
            clipped_action = self.actor.unscale(clipped_action)
        return action.unsqueeze(0), clipped_action.unsqueeze(0), rhs, {}
    
    def training_step(self, batch) -> torch.Tensor:
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
        prediction = self.actor.dynamics_model(train_inputs)
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
