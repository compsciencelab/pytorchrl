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
    """[summary]

    Args:
        Algorithm ([type]): [description]
    """
    def __init__(self,
                 actor,
                 device,
                 config,
                 ):

        # ---- General algo attributes ----------------------------------------

        # Number of steps collected with initial random policy
        self._start_steps = int(config.start_steps)

        # Times data in the buffer is re-used before data collection proceeds
        self._num_epochs = int(1)  # Default to 1 for off-policy algorithms


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

        self.action_high = torch.from_numpy(self.mpc.action_high).float().to(device)
        self.action_low = torch.from_numpy(self.mpc.action_low).float().to(device)

        self.iter = 0
        self.device = device
        self.max_grad_norm = 0.5
        self._update_every = config.update_every
        self.reuse_data = False
        
        # training break conditions
        self.mb_train_epochs = 0
        self.max_not_improvements = min(round((self._start_steps + self._update_every) / self._mini_batch_size, 0), 5)
        self._current_best = [1e10 for i in range(self.actor.ensemble_size)]
        self.improvement_threshold = 0.01
        self.break_counter = 0

        # ----- Optimizers ----------------------------------------------------
        self.dynamics_optimizer = optim.Adam(self.actor.dynamics_model.parameters(), lr=config.lr)

    @classmethod
    def create_factory(cls,
                       config,
                       ):

        def create_algo_instance(device, actor):
            return cls(actor=actor,
                       device=device,
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
        with torch.no_grad():
            action = self.mpc.get_action(state=obs, model=self.actor, noise=False)
            clipped_action = torch.clamp(action, -1, 1)
            
        if self.actor.unscale:
            action = self.actor.unscale(action)
            clipped_action = self.actor.unscale(clipped_action)
        return clipped_action.unsqueeze(-1), clipped_action.unsqueeze(-1), rhs, {}
    
    
    def training_step(self, batch)-> Tuple[torch.Tensor, torch.Tensor]:
        train_inputs = batch["train_input"]
        train_labels = batch["train_label"]

        self.actor.train()
        mean = self.actor.dynamics_model(inputs=train_inputs)
        loss = self.actor.calculate_loss(mean=mean,
                                         labels=train_labels)
        return loss
    
    def validation(self, batch)-> Tuple[torch.Tensor, bool]:
        holdout_inputs = batch["holdout_inputs"]
        holdout_labels = batch["holdout_labels"]
        self.actor.eval()
        with torch.no_grad():
            val_mean = self.actor.dynamics_model(inputs=holdout_inputs)
            validation_loss = self.actor.calculate_loss(mean=val_mean,
                                                        labels=holdout_labels,
                                                        validate=True)

            validation_loss = validation_loss.cpu().numpy()
            assert validation_loss.shape == (self.actor.ensemble_size, )
            sorted_loss_idx = np.argsort(validation_loss)
            self.actor.elite_idxs = sorted_loss_idx[:self.actor.elite_size].tolist()
            break_condition = self.test_break_condition(validation_loss)
        
        return validation_loss.mean(), break_condition
    
    def test_break_condition(self, current_losses):
        keep_train = False
        for i in range(len(current_losses)):
            current_loss = current_losses[i]
            best_loss = self._current_best[i]
            improvement = (best_loss - current_loss) / best_loss
            if improvement > self.improvement_threshold:
                self._current_best[i] = current_loss
                keep_train = True
    
        if keep_train:
            self.break_counter = 0
        else:
            self.break_counter += 1
        if self.break_counter >= self.max_not_improvements:
            return True
        else:
            return False
    
    

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
        if batch["batch_number"] == 0:
            self.reuse_data = True
            self.mb_train_epochs += 1

        train_loss = self.training_step(batch)

        self.dynamics_optimizer.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.dynamics_model.parameters(), self.max_grad_norm)
        dyna_grads = get_gradients(self.actor.dynamics_model, grads_to_cpu=grads_to_cpu)

        info = {"train_loss": train_loss.item()}

        # Validation run 
        if batch["batch_number"] == batch["max_batches"]:
            validation_loss, break_condition = self.validation(batch)
            info.update({"validation_loss": validation_loss.item()})

            if break_condition:
                self.reuse_data = False
                info.update({"Training Epoch": self.mb_train_epochs})
                self.mb_train_epochs = 0
                self.break_counter = 0

        grads = {"dyna_grads": dyna_grads}

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
