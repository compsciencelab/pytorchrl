from numpy.lib.mixins import NDArrayOperatorsMixin
from typing import Tuple

import gym
import torch
import time
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import random
from torch.nn.functional import one_hot


import pytorchrl as prl
from pytorchrl.agent.actors.distributions import get_dist
from pytorchrl.agent.actors.reward_functions import get_reward_function
from pytorchrl.agent.actors.utils import Scale, Unscale, init, partially_load_checkpoint
from pytorchrl.agent.actors.feature_extractors.ensemble_layer import EnsembleFC
from pytorchrl.agent.actors.base import Actor


class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.
        Arguments:
        data (np.ndarray): A numpy array containing the input
        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu


class MBActor(Actor):
    """
    Model-Based Actor class for Model-Based algorithms.

    It contains the dynamics network to predict the next state (and reward if selected). 

    Parameters
    ----------
    env_id: str
        Name of the gym environment
    input_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space.
    hidden_size: int
        Hidden size number.
    hidden_layer: int
        Number of hidden layers.
    batch_size: int
        Batch size.
    ensemble_size: int
        Number of models in the ensemble.
    elite_size: int
        Number of the elite members of the ensemble.
    dynamics_type: str
        Type of the dynamics module. Can be probabilistic or deterministic.
    learn_reward_function: int
        Either 0 or 1 if the reward function should be learned (1) or will be provided (0).
    device: torch.device
        CPU or specific GPU where class computations will take place.
    checkpoint : str
        Path to a previously trained Actor checkpoint to be loaded.
    """
    def __init__(self,
                 env_id,
                 input_space,
                 action_space,
                 hidden_size,
                 hidden_layer,
                 batch_size,
                 ensemble_size,
                 elite_size,
                 dynamics_type,
                 learn_reward_function,
                 device,
                 checkpoint)-> None:
        super(MBActor, self).__init__(device=device,
                                      checkpoint=checkpoint,
                                      input_space=input_space,
                                      action_space=action_space)
        self.device = device
        self.input_space = input_space.shape[0]
        self.reward_function = None
      
        if learn_reward_function == 0:
            self.reward_function = get_reward_function(env_id=env_id)
            self.predict = self.predict_given_reward
        else:
            self.predict = self.predict_learned_reward
        self.ensemble_size = ensemble_size
        self.elite_size = elite_size
        self.elite_idxs = [i for i in range(self.elite_size)]

        # Scaler for scaling training inputs
        self.standard_scaler = StandardScaler()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.dynamics_type = dynamics_type
        assert dynamics_type in ["probabilistic", "deterministic"]

        self.create_dynamics()
        print(self.dynamics_model)

    @classmethod
    def create_factory(
            cls,
            env_id,
            input_space,
            action_space,
            hidden_size,
            hidden_layer,
            batch_size,
            ensemble_size,
            elite_size,
            dynamics_type="probabilistic",
            learn_reward_function=False,
            checkpoint=None):
        """[summary]


        Parameters
        ----------
        env_id: str
            Name of the gym environment
        input_space : gym.Space
            Environment observation space.
        action_space : gym.Space
            Environment action space.
        hidden_size: int
            Hidden size number.
        hidden_layer: int
            Number of hidden layers.
        batch_size: int
            Batch size.
        ensemble_size: int
            Number of models in the ensemble.
        elite_size: int
            Number of the elite members of the ensemble.
        dynamics_type: str
            Type of the dynamics module. Can be probabilistic or deterministic.
        learn_reward_function: int
            Either 0 or 1 if the reward function should be learned (1) or will be provided (0).
        checkpoint : str
            Path to a previously trained Actor checkpoint to be loaded.

        Returns
        -------
        create_dynamics_instance : func
            creates a new dynamics model class instance.
        """

        def create_dynamics_instance(device):
            """Create and return an dynamics model instance."""
            model = cls(env_id=env_id,
                         input_space=input_space,
                         action_space=action_space,
                         hidden_size=hidden_size,
                         hidden_layer=hidden_layer,
                         batch_size=batch_size,
                         ensemble_size=ensemble_size,
                         elite_size=elite_size,
                         dynamics_type=dynamics_type,
                         learn_reward_function=learn_reward_function,
                         checkpoint=checkpoint,
                         device=device)
            model.to(device)

            try:
                model.try_load_from_checkpoint()
            except RuntimeError:
                pass

            return model

        return create_dynamics_instance

    def get_action(self,):
        pass
    
    def is_recurrent(self,):
        return False
    
    def recurrent_hidden_state_size(self):
        return 0

    def actor_initial_states(self, obs):
        """
        Returns all actor inputs required to predict initial action.

        Parameters
        ----------
        obs : torch.tensor
            Initial environment observation.

        Returns
        -------
        obs : torch.tensor
            Initial environment observation.
        rhs : dict
            Initial recurrent hidden state (will contain zeroes).
        done : torch.tensor
            Initial done tensor, indicating the environment is not done.
        """

        if isinstance(obs, dict):
            num_proc = list(obs.values())[0].size(0)
            dev = list(obs.values())[0].device
        else:
            num_proc = obs.size(0)
            dev = obs.device

        done = torch.zeros(num_proc, 1).to(dev)
        rhs_act = torch.zeros(num_proc, 42).to(dev)

        rhs = {"rhs_act": rhs_act}
        rhs.update({"rhs_q{}".format(i + 1): rhs_act.clone() for i in range(1)})

        return obs, rhs, done


    def create_dynamics(self, name="dynamics_model"):
        """
        Create a dynamics model and define it as class attribute under the name `name`.

        Parameters
        ----------
        name : str
            dynamics model name.
        """
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            input_layer = EnsembleFC(self.input_space + self.action_space.n, out_features=self.hidden_size, ensemble_size=self.ensemble_size)
        else:
            input_layer = EnsembleFC(self.input_space + self.action_space.shape[0], out_features=self.hidden_size, ensemble_size=self.ensemble_size)

        dynamics_layers = []
        dynamics_layers.append(input_layer)
        dynamics_layers.append(nn.SiLU())

        for _ in range(self.hidden_layer):
            dynamics_layers.append(EnsembleFC(self.hidden_size, self.hidden_size, self.ensemble_size))
            dynamics_layers.append(nn.SiLU())
        
        if self.reward_function is None:
            num_outputs = self.input_space + 1
        else:
            num_outputs = self.input_space

        if self.dynamics_type == "deterministic":
            output_layer = get_dist("DeterministicEnsemble")(num_inputs=self.hidden_size,
                                                       num_outputs=num_outputs,
                                                       ensemble_size=self.ensemble_size)
        elif self.dynamics_type == "probabilistic":
            output_layer = get_dist("DiagGaussianEnsemble")(num_inputs=self.hidden_size,
                                                      num_outputs=num_outputs,
                                                      ensemble_size=self.ensemble_size)
        else:
            raise ValueError
        dynamics_layers.append(output_layer)
        
        if type(self.action_space) == gym.spaces.box.Box:
            self.scale = Scale(self.action_space)
            self.unscale = Unscale(self.action_space)
        else:
            self.scale = None
            self.unscale = None

        dynamics_net = nn.Sequential(*dynamics_layers)

        setattr(self, name, dynamics_net)

    def check_dynamics_weights(self, parameter1, parameter2):
        for p1, p2 in zip(parameter1, parameter2):
             if p1.data.ne(p2.data).sum() > 0:
                 return False
        return True

    def reinitialize_dynamics_model(self, ):
        old_weights = self.dynamics_model.parameters()
        self.create_dynamics()
        self.dynamics_model.to(self.device)
        new_weights = self.dynamics_model.parameters()
        assert not self.check_dynamics_weights(old_weights, new_weights)
    
    def predict_learned_reward(self, states: torch.Tensor, actions: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """Does the next state prediction and reward prediction with a learn reward function.

        Parameters
        ----------
            states (torch.Tensor): Current state s
            actions (torch.Tensor): Action taken in state s

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the next state and reward prediction.
        """
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            actions = one_hot(actions, num_classes=self.action_space.n).squeeze(1)

        inputs = torch.cat((states, actions), dim=-1)
        inputs = inputs[None, :, :].repeat(self.ensemble_size, 1, 1).float() # [ensemble size, batch size, input size]

        ensemble_means = self.dynamics_model(inputs)
        ensemble_means[:, :, :-1] += states.to(self.device)
        elite_mean = ensemble_means[self.elite_idxs]
        
        assert elite_mean.shape == (self.elite_size, states.shape[0], states.shape[1]+1)

        if self.dynamics_type == "probabilistic":
            mean_predictions = torch.normal(mean=elite_mean, std=0.01)
            predictions = mean_predictions.mean(0)
        else:
            predictions = elite_mean.mean(0)

        assert predictions.shape == (states.shape[0], states.shape[1] + 1)

        next_states = predictions[:, :-1]
        rewards = predictions[:, -1].unsqueeze(-1)
        # TODO: add Termination function?
        return next_states, rewards
    
    def predict_given_reward(self, states: torch.Tensor, actions: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """Does the next state prediction and calculates the reward given a reward function. 

        Parameters
        ----------
            states (torch.Tensor): Current state s
            actions (torch.Tensor): Action taken in state s

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the next state and calculated reward.
        """
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            actions = one_hot(actions, num_classes=self.action_space.n).squeeze(1)

        inputs = torch.cat((states, actions), dim=-1)
        inputs = inputs[None, :, :].repeat(self.ensemble_size, 1, 1).float() # [ensemble size, batch size, input size]

        ensemble_means = self.dynamics_model(inputs)
        ensemble_means += states.to(self.device)
        elite_mean = ensemble_means[self.elite_idxs]
        
        assert elite_mean.shape == (self.elite_size, states.shape[0], states.shape[1])

        if self.dynamics_type == "probabilistic":
            mean_predictions = torch.normal(mean=elite_mean, std=0.01)
            predictions = mean_predictions.mean(0)
        else:
            predictions = elite_mean.mean(0)

        assert predictions.shape == (states.shape[0], states.shape[1])

        next_states = predictions

        rewards = self.reward_function(states, actions)
        # TODO: add Termination function?
        return next_states, rewards


    def do_rollout(self, state, action):
        raise NotImplementedError


    def calculate_loss(self, mean: torch.Tensor,
                             labels: torch.Tensor,
                             validate: bool=False
                             )-> torch.Tensor:
        """Calculate the MSE loss.

        Args:
            mean (torch.Tensor): Mean prediction of the next state
            labels (torch.Tensor): Training labels
            validate (bool, optional): Set to True if calculating the mse errors for each ensemble member. Defaults to False.

        Returns:
            torch.Tensor: MSE loss
        """

        if not validate:
            return ((mean - labels)**2).mean(-1).mean(-1).sum()
        else:
            mse_loss = ((mean - labels)**2).mean(-1).mean(-1)
            return mse_loss

