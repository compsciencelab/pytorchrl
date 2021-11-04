from numpy.lib.mixins import NDArrayOperatorsMixin
from typing import Tuple

import gym
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import random


import pytorchrl as prl
from pytorchrl.agent.actors.distributions import get_dist
from pytorchrl.agent.actors.utils import Scale, Unscale, init, partially_load_checkpoint
from pytorchrl.agent.actors.feature_extractors.ensemble_layer import EnsembleFC


class StandardScaler(object):
    def __init__(self)-> None:
        self.mu = np.zeros(1)
        self.std = np.ones(1)

    def fit(self, data: np.array)-> None:
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data: np.array)-> np.array:
        return (data - self.mu) / self.std

    def inverse_transform(self, data: np.array)-> np.array:
        return self.std * data + self.mu


class MBActor(nn.Module):
    def __init__(self,
                 input_space,
                 action_space,
                 ensemble_size,
                 elite_size,
                 dynamics_type,
                 device,
                 noise=None)-> None:
        super(MBActor, self).__init__()

        self.noise = noise

        self.device = device
        self.input_space = input_space.shape[0]
        self.action_space = action_space
        self.ensemble_size = ensemble_size
        self.elite_size = elite_size
        self.elite_idxs = [i for i in range(self.elite_size)]
        self.scaler = StandardScaler()
        self.rollout_select = "random"
        self.batch_size = 256
        self.hidden_size = 200
        self.dynamics_type = dynamics_type
        assert dynamics_type in ["probabilistic", "deterministic"]

        self.create_dynamics()

    @classmethod
    def create_factory(
            cls,
            input_space,
            action_space,
            ensemble_size,
            elite_size,
            dynamics_type="probabilistic",
            restart_model=None):

        def create_dynamics_instance(device):
            """Create and return an actor critic instance."""
            model = cls(input_space=input_space,
                         action_space=action_space,
                         ensemble_size=ensemble_size,
                         elite_size=elite_size,
                         dynamics_type=dynamics_type,
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

    def get_action(self, obs, deterministic=False):
        return 

    def create_dynamics(self, name="dynamics_model"):
        dynamics_layer_1 = EnsembleFC(self.input_space + self.action_space.shape[0], out_features=self.hidden_size, ensemble_size=self.ensemble_size)
        dynamics_layer_2 = EnsembleFC(in_features=self.hidden_size, out_features=self.hidden_size, ensemble_size=self.ensemble_size)
        dynamics_layer_3 = EnsembleFC(in_features=self.hidden_size, out_features=self.hidden_size, ensemble_size=self.ensemble_size)
        dynamics_layer_4 = EnsembleFC(in_features=self.hidden_size, out_features=self.hidden_size, ensemble_size=self.ensemble_size)

        if self.dynamics_type == "deterministic":
            output = get_dist("DeterministicEnsemble")(num_inputs=self.hidden_size,
                                                       num_outputs=self.input_space + 1,
                                                       ensemble_size=self.ensemble_size)
        elif self.dynamics_type == "probabilistic":
            output = get_dist("DiagGaussianEnsemble")(num_inputs=self.hidden_size,
                                                      num_outputs=self.input_space + 1,
                                                      ensemble_size=self.ensemble_size)
        else:
            raise ValueError
        
        self.scale = Scale(self.action_space)
        self.unscale = Unscale(self.action_space)

        # ---- 6. Concatenate all dynamics net modules ------------------------------
        dynamics_net = nn.Sequential(OrderedDict([
            ('dynamics_layer_1', dynamics_layer_1),
            ('swish1', nn.SiLU()),
            ('dynamics_layer_2', dynamics_layer_2),
            ('swish2', nn.SiLU()),
            ('dynamics_layer_3', dynamics_layer_3),
            ('swish3', nn.SiLU()),
            ('dynamics_layer_4', dynamics_layer_4),
            ('swish4', nn.SiLU()),
            ("output", output),
        ]))

        setattr(self, name, dynamics_net)

    def get_prediction(self,
                       inputs: torch.Tensor,
                       ret_log_var: bool=False
                       )-> Tuple[torch.Tensor, torch.Tensor]:

        if ret_log_var:
            mean, log_var, min_max_var = self.dynamics_model(inputs)
            return mean, log_var, min_max_var
        else:
            mean, var, min_max_var = self.dynamics_model(inputs)
            return mean, var, min_max_var

    def predict(self, states: torch.Tensor, actions: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((states, actions), dim=-1)
        # TODO: fix this torch -> numpy -> torch // cuda -> cpu -> cuda 
        inputs = torch.from_numpy(self.scaler.transform(inputs.cpu().numpy())).float().to(self.device)
        inputs = inputs[None, :, :].repeat(self.ensemble_size, 1, 1)
        ensemble_means, ensemble_var, _ = self.get_prediction(inputs=inputs, ret_log_var=False)

        ensemble_means[:, :, :-1] += states.to(self.device)
        ensemble_std = torch.sqrt(ensemble_var)
        if self.dynamics_type == "probabilisitc":
            ensemble_predictions = torch.normal(ensemble_means, ensemble_std)
        else:
            ensemble_predictions = ensemble_means
        if self.rollout_select == "random":
            # choose what predictions we select from what ensemble member
            ensemble_idx = random.choices(self.elite_idxs, k=states.shape[0])
            step_idx = np.arange(states.shape[0])
            # pick prediction based on ensemble idxs
            predictions = ensemble_predictions[ensemble_idx, step_idx, :]
        else:
            predictions = ensemble_predictions[self.elite_idxs].mean(0)
        assert predictions.shape == (states.shape[0], states.shape[1] + 1)

        next_states = predictions[:, :-1]
        rewards = predictions[:, -1].unsqueeze(-1)
        # TODO: add Termination function?
        return next_states, rewards

    def do_rollout(self, state, action):
        raise NotImplementedError

    def calculate_loss(self, mean: torch.Tensor,
                             logvar: torch.Tensor,
                             min_max_var: Tuple[torch.Tensor, torch.Tensor],
                             labels: torch.Tensor,
                             inc_var_loss: bool=True
                             )-> Tuple[torch.Tensor, torch.Tensor]:

        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
            total_loss += 0.01 * torch.sum(min_max_var[1]) - 0.01 * torch.sum(min_max_var[0])
            return total_loss, mse_loss
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
            return total_loss

    def training_step(self, batch)-> torch.Tensor:
        train_inputs = batch["train_input"]
        train_labels = batch["train_label"]

        holdout_inputs = batch["holdout_inputs"]
        holdout_labels = batch["holdout_labels"]
        
        self.train()
        mean, log_var, min_max_var = self.get_prediction(inputs=train_inputs, ret_log_var=True)
        loss, _ = self.calculate_loss(mean=mean, logvar=log_var, min_max_var=min_max_var, labels=train_labels, inc_var_loss=True)
        
        self.eval()
        with torch.no_grad():
            val_mean, val_log_var, _ = self.get_prediction(inputs=holdout_inputs, ret_log_var=True)
            validation_loss = self.calculate_loss(mean=val_mean, logvar=val_log_var, min_max_var=min_max_var, labels=holdout_labels, inc_var_loss=False)
            validation_loss = validation_loss.detach().cpu().numpy()
            sorted_loss_idx = np.argsort(validation_loss)
            self.elite_idxs = sorted_loss_idx[:self.elite_size].tolist()
            # TODO: add early stopping

        return loss
