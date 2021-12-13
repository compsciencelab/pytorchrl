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


class MBActor(nn.Module):
    def __init__(self,
                 env_id,
                 input_space,
                 action_space,
                 ensemble_size,
                 elite_size,
                 dynamics_type,
                 learn_reward_function,
                 device,
                 noise=None)-> None:
        super(MBActor, self).__init__()

        self.noise = noise

        self.device = device
        self.input_space = input_space.shape[0]
        self.action_space = action_space
        self.reward_function = None
      
        if learn_reward_function == 0:
            self.reward_function = get_reward_function(env_id=env_id)
            self.predict = self.predict_given_reward
        else:
            self.predict = self.predict_learned_reward
        self.ensemble_size = ensemble_size
        self.elite_size = elite_size
        self.elite_idxs = [i for i in range(self.elite_size)]

        
        self.batch_size = 256
        self.hidden_size = 200
        self.hidden_layer = 3
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
            ensemble_size,
            elite_size,
            dynamics_type="probabilistic",
            learn_reward_function=False,
            restart_model=None):

        def create_dynamics_instance(device):
            """Create and return an actor critic instance."""
            model = cls(env_id=env_id,
                         input_space=input_space,
                         action_space=action_space,
                         ensemble_size=ensemble_size,
                         elite_size=elite_size,
                         dynamics_type=dynamics_type,
                         learn_reward_function=learn_reward_function,
                         device=device)

            if isinstance(restart_model, str):
                model.load_state_dict(torch.load(restart_model, map_location=device))
            elif isinstance(restart_model, dict):
                for submodule, checkpoint in restart_model.items():
                    partially_load_checkpoint(model, submodule, checkpoint)
            model.to(device)

            return model

        return create_dynamics_instance


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


    def get_prediction(self,
                       inputs: torch.Tensor,
                       ret_log_var: bool=False
                       )-> Tuple[torch.Tensor, torch.Tensor]:
        # inputs: [batch size, state size + action size]
        mean = self.dynamics_model(inputs) #  log_var, min_max_var

        if ret_log_var:
            return mean # , log_var, min_max_var
        else:
            return mean #, torch.exp(log_var), min_max_var


    def predict_learned_reward(self, states: torch.Tensor, actions: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:

        if type(self.action_space) == gym.spaces.discrete.Discrete:
            actions = one_hot(actions, num_classes=self.action_space.n).squeeze(1)

        inputs = torch.cat((states, actions), dim=-1)
        inputs = inputs[None, :, :].repeat(self.ensemble_size, 1, 1).float() # [ensemble size, batch size, input size]

        ensemble_means = self.get_prediction(inputs=inputs, ret_log_var=False) #, ensemble_var, _
        ensemble_means[:, :, :-1] += states.to(self.device)
        elite_mean = ensemble_means[self.elite_idxs]
        #elite_std = ensemble_var[self.elite_idxs]
        
        assert elite_mean.shape == (self.elite_size, states.shape[0], states.shape[1]+1)
        #assert elite_std.shape == (self.elite_size, states.shape[0], states.shape[1]+1)


        if self.dynamics_type == "probabilistic":
            mean_predictions = torch.normal(mean=elite_mean) #, std=stds)
            predictions = mean_predictions.mean(0)
        else:
            predictions = elite_mean.mean(0)

        assert predictions.shape == (states.shape[0], states.shape[1] + 1)

        next_states = predictions[:, :-1]
        rewards = predictions[:, -1].unsqueeze(-1)
        # TODO: add Termination function?
        return next_states, rewards
    
    def predict_given_reward(self, states: torch.Tensor, actions: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        if type(self.action_space) == gym.spaces.discrete.Discrete:
            actions = one_hot(actions, num_classes=self.action_space.n).squeeze(1)

        inputs = torch.cat((states, actions), dim=-1)
        inputs = inputs[None, :, :].repeat(self.ensemble_size, 1, 1).float() # [ensemble size, batch size, input size]

        ensemble_means = self.get_prediction(inputs=inputs, ret_log_var=False) # , ensemble_var, _
        ensemble_means += states.to(self.device)
        elite_mean = ensemble_means[self.elite_idxs]
        #elite_std = ensemble_var[self.elite_idxs]
        
        assert elite_mean.shape == (self.elite_size, states.shape[0], states.shape[1])
        #assert elite_std.shape == (self.elite_size, states.shape[0], states.shape[1])

        #stds = torch.sqrt(elite_std).mean(0)    

        if self.dynamics_type == "probabilistic":
            mean_predictions = torch.normal(mean=elite_mean) #, std=stds)
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
                             logvar: torch.Tensor,
                             min_max_var: Tuple[torch.Tensor, torch.Tensor],
                             labels: torch.Tensor,
                             validate: bool=False
                             )-> Tuple[torch.Tensor, torch.Tensor]:

        
        # inv_var = (-logvar).exp()
        if not validate:
            return ((mean - labels)**2).mean(-1).mean(-1).sum()
        #     total_loss = ((mean - labels)**2 * inv_var).mean(-1).mean(-1).sum() + logvar.mean(-1).mean(-1).sum()
        #     total_loss = total_loss + 0.01 * torch.sum(min_max_var[1]) - 0.01 * torch.sum(min_max_var[0])
        #     return total_loss
        else:
            mse_loss = ((mean - labels)**2).mean(-1).mean(-1)
            return mse_loss

