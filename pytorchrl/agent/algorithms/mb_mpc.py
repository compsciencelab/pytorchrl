import itertools
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
        self._num_epochs = int(10)  # Default to 1 for off-policy algorithms


        # Size of update mini batches
        self._mini_batch_size = int(config.mini_batch_size)
        self._num_mini_batch = 1
        # Number of network updates between test evaluations
        self._test_every = int(5000)

        # Number of episodes to complete when testing
        self._num_test_episodes = int(5)
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
        self.reuse_data = True
        
        # training break conditions
        self.num_epochs = 0
        self.max_not_improvements = 5
        self._current_best = [1e10 for i in range(self.actor.ensemble_size)]
        self.improvement_threshold = 0.01
        self.break_counter = 0

        # List of parameters for the dynamics Model
        dynamics_params = itertools.chain(self.actor.dynamics_model.parameters())

        # ----- Optimizers ----------------------------------------------------

        self.dynamics_optimizer = optim.Adam(dynamics_params, lr=config.lr)

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
        # do MPC planning here just pass the model in there
        with torch.no_grad():
            action = self.mpc.get_action(state=obs, model=self.actor, noise=self.action_noise)
            clipped_action = torch.clamp(action, -1, 1)
        if self.actor.unscale:
            action = self.actor.unscale(action)
            clipped_action = self.actor.unscale(clipped_action)

        return action.unsqueeze(-1), clipped_action.unsqueeze(-1), rhs, {}
    
    
    def training_step(self, batch)-> torch.Tensor:
        train_inputs = batch["train_input"]
        train_labels = batch["train_label"]

        holdout_inputs = batch["holdout_inputs"]
        holdout_labels = batch["holdout_labels"]
        
        self.actor.train()
        mean, logvar, min_max_var = self.actor.get_prediction(inputs=train_inputs, ret_log_var=True)
        loss, total_loss_min_max = self.actor.calculate_loss(mean=mean,
                                                       logvar=logvar,
                                                       min_max_var=min_max_var,
                                                       labels=train_labels,
                                                       inc_var_loss=True)
        
        self.actor.eval()
        with torch.no_grad():
            val_mean, val_log_var, _ = self.actor.get_prediction(inputs=holdout_inputs, ret_log_var=True)
            validation_loss = self.actor.calculate_loss(mean=val_mean,
                                                  logvar=val_log_var,
                                                  min_max_var=min_max_var,
                                                  labels=holdout_labels,
                                                  inc_var_loss=False)
            validation_loss = validation_loss.detach().cpu().numpy()
            sorted_loss_idx = np.argsort(validation_loss)
            self.elite_idxs = sorted_loss_idx[:self.actor.elite_size].tolist()
            break_condition = # self.test_break_condition(validation_loss)

        return loss, total_loss_min_max, validation_loss, break_condition
    
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
        if batch["batch_number"] == 1:
            self.reuse_data = True
            self.num_epochs += 1
        logging_loss, train_loss, validation_loss, break_condition = self.training_step(batch)
        self.dynamics_optimizer.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.dynamics_model.parameters(), self.max_grad_norm)
        dyna_grads = get_gradients(self.actor.dynamics_model, grads_to_cpu=grads_to_cpu)

        info = {
            "train_loss": logging_loss.item(),
            "validation_loss": validation_loss.item(),
            "Training Epoch": self.num_epochs
        }
        if break_condition:
            self.reuse_data = False
            self.num_epochs = 0

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
