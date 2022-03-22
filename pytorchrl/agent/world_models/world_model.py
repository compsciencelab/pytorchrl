from typing import Tuple

import gym
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from pytorchrl.agent.actors.distributions import get_dist
from pytorchrl.agent.actors.reward_functions import get_reward_function
from pytorchrl.agent.actors.utils import Scale, Unscale
from pytorchrl.agent.actors.base import Actor


class StandardScaler(object):
    def __init__(self, device):
        self.input_mu = torch.zeros(1).to(device)
        self.input_std = torch.ones(1).to(device)
        self.target_mu = torch.zeros(1).to(device)
        self.target_std = torch.ones(1).to(device)
        self.device = device

    def fit(self, inputs, targets):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.
        Arguments:
        inputs (torch.Tensor): A torch Tensor containing the input
        targets (torch.Tensor): A torch Tensor containing the input
        Returns: None.
        """
        self.input_mu = torch.mean(inputs, dim=0, keepdims=True).to(self.device)
        self.input_std = torch.std(inputs, dim=0, keepdims=True).to(self.device)
        self.input_std[self.input_std < 1e-8] = 1.0
        self.target_mu = torch.mean(targets, dim=0, keepdims=True).to(self.device)
        self.target_std = torch.std(targets, dim=0, keepdims=True).to(self.device)
        self.target_std[self.target_std < 1e-8] = 1.0

    def transform(self, inputs, targets=None):
        """Transforms the input matrix data using the parameters of this scaler.
        Arguments:
        inputs (torch.Tensor): A torch Tensor containing the points to be transformed.
        targets (torch.Tensor): A torch Tensor containing the points to be transformed.
        Returns: (torch.Tensor, torch.Tensor) The transformed datasets.
        """
        norm_inputs = (inputs - self.input_mu) / self.input_std
        norm_targets = None
        if targets != None:
            norm_targets = (targets - self.target_mu) / self.target_std
        return norm_inputs, norm_targets

    def inverse_transform(self, targets):
        """Undoes the transformation performed by this scaler.
        Arguments:
        targets (torch.Tensor): A torch Tensor containing the points to be transformed.
        Returns: (torch.Tensor) The transformed dataset.
        """
        return self.target_std * targets + self.target_mu


class WorldModel(Actor):
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
    batch_size: int
        Batch size.
    learn_reward_function: bool
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
                 batch_size,
                 learn_reward_function,
                 device,
                 checkpoint) -> None:
        super(WorldModel, self).__init__(device=device,
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

        # Scaler for scaling training inputs
        self.standard_scaler = StandardScaler(device)
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.create_dynamics()

    @classmethod
    def create_factory(
            cls,
            env_id,
            input_space,
            action_space,
            hidden_size,
            batch_size,
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
        batch_size: int
            Batch size.
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
                         batch_size=batch_size,
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

    def get_action(self):
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
            input_layer = nn.Linear(self.input_space + self.action_space.n, out_features=self.hidden_size)
        else:
            input_layer = nn.Linear(self.input_space + self.action_space.shape[0], out_features=self.hidden_size)

        hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        
        if self.reward_function is None:
            num_outputs = self.input_space + 1
        else:
            num_outputs = self.input_space

        output_layer = get_dist("DeterministicMB")(num_inputs=self.hidden_size, num_outputs=num_outputs)
            
        self.activation = nn.ReLU()

        dynamics_layers = [input_layer,
                           self.activation,
                           hidden_layer,
                           self.activation,
                           output_layer]
        
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
        """Reinitializes the dynamics model, can be done before each new Model learning run.
           Might help in some environments to overcome overfitting of the model!
        """
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

        # scale inputs based on recent batch scalings
        norm_inputs, _ = self.standard_scaler.transform(inputs)
        norm_predictions = self.dynamics_model(norm_inputs)

        # inverse transform outputs
        predictions = self.standard_scaler.inverse_transform(norm_predictions)
        predictions[:, :-1] += states.to(self.device)

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

        # scale inputs based on recent batch scalings
        norm_inputs, _ = self.standard_scaler.transform(inputs)

        norm_predictions = self.dynamics_model(norm_inputs)

        # inverse transform outputs
        predictions = self.standard_scaler.inverse_transform(norm_predictions)
        predictions += states.to(self.device)
        next_states = predictions

        rewards = self.reward_function(states, actions, next_states)
        # TODO: add Termination function?
        return next_states, rewards

    def do_rollout(self, state, action):
        raise NotImplementedError


