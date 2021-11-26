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
                #  n_planner=5000,
                #  planning_depth=32,
                #  update_every=50,
                #  test_every=1000,
                #  max_grad_norm=0.5,
                #  start_steps=20000,
                #  mini_batch_size=256,
                #  num_test_episodes=5
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
        self._test_every = int(config.test_every)

        # Number of episodes to complete when testing
        self._num_test_episodes = int(config.num_test_episodes)
        self.actor = actor
        self.action_noise = config.action_noise
        # ---- MB MPC-specific attributes ----------------------------------------
        if config.mpc_type == "RS":
            self.mpc = MPC.RandomShooting(action_space=self.actor.action_space,
                            n_planner=config.n_planner,
                            horion=config.horizon,
                            device=device)
        elif config.mpc_type == "CEM":
            self.mpc = MPC.CEM(action_space=self.actor.action_space,
                            n_planner=config.n_planner,
                            horion=config.horizon,
                            device=device)
        elif config.mpc_type == "PDDM":
            self.mpc = MPC.PDDM(action_space=self.actor.action_space,
                n_planner=config.n_planner,
                horion=config.horizon,
                device=device)
        else:
            raise ValueError
        self.iter = 0
        self.device = device
        self.max_grad_norm = config.max_grad_norm
        self._update_every = config.update_every

        # List of parameters for the dynamics Model
        dynamics_params = itertools.chain(self.actor.dynamics_model.parameters())

        # ----- Optimizers ----------------------------------------------------

        self.dynamics_optimizer = optim.Adam(dynamics_params, lr=config.lr)

    @classmethod
    def create_factory(cls,
                       config,
                    #    n_planner=5000,
                    #    planning_depth=32,
                    #    test_every=5000,
                    #    update_every=50,
                    #    start_steps=1000,
                    #    max_grad_norm=0.5,
                    #    mini_batch_size=256,
                    #    num_test_episodes=5
                       ):

        def create_algo_instance(device, actor):
            return cls(actor=actor,
                       device=device,
                       config=config,)
                    #    n_planner=n_planner,
                    #    planning_depth=planning_depth,
                    #    test_every=test_every,
                    #    start_steps=start_steps,
                    #    update_every=update_every,
                    #    max_grad_norm=max_grad_norm,
                    #    mini_batch_size=mini_batch_size,
                    #    num_test_episodes=num_test_episodes)

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

        logging_loss, train_loss, validation_loss = self.actor.training_step(batch)
        self.dynamics_optimizer.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.dynamics_model.parameters(), self.max_grad_norm)
        dyna_grads = get_gradients(self.actor.dynamics_model, grads_to_cpu=grads_to_cpu)

        info = {
            "train_loss": logging_loss.item(),
            "validation_loss": validation_loss.item()
        }

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
