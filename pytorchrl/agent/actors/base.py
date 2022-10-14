import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pytorchrl.agent.actors.utils import partially_load_checkpoint


class Actor(nn.Module, ABC):
    """
    Base class for all Actors.

    Parameters
    ----------
    device: torch.device
        CPU or specific GPU where class computations will take place.
    input_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space.
    checkpoint : str
        Path to a previously trained Actor checkpoint to be loaded.
    """
    def __init__(self,
                 device,
                 input_space,
                 action_space,
                 checkpoint=None,
                 *args):

        super(Actor, self).__init__()
        self.device = device
        self.checkpoint = checkpoint
        self.input_space = input_space
        self.action_space = action_space

    @classmethod
    @abstractmethod
    def create_factory(
            cls,
            device,
            input_space,
            action_space,
            checkpoint=None,
            *args):
        """
        Returns a function that creates actor critic instances.

        Parameters
        ----------
        device: torch.device
            CPU or specific GPU where class computations will take place.
        input_space : gym.Space
            Environment observation space.
        action_space : gym.Space
            Environment action space.
        checkpoint : str
            Path to a previously trained Actor checkpoint to be loaded.

        Returns
        -------
        create_actor_instance : func
            creates a new Actor class instance.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_recurrent(self, *args):
        """Returns True if the actor network are recurrent."""
        raise NotImplementedError

    @property
    @abstractmethod
    def recurrent_hidden_state_size(self):
        """Size of policy recurrent hidden state"""
        raise NotImplementedError

    @abstractmethod
    def actor_initial_states(self, obs, *args):
        """
        Returns all policy inputs to predict the environment initial action.

        Parameters
        ----------
        obs : torch.tensor
            Initial environment observation.

        Returns
        -------
        obs : torch.tensor
            Initial environment observation.
        rhs : torch.tensor
            Initial recurrent hidden state.
        done : torch.tensor
            Initial done tensor, indicating the environment is not done.
        """
        raise NotImplementedError

    def try_load_from_checkpoint(self):
        """Load weights from previously saved checkpoint."""
        try:
            if isinstance(self.checkpoint, str):
                print("Loading all model weight from {}".format(self.checkpoint))
                self.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
            elif isinstance(self.checkpoint, dict):
                for submodule, checkpoint in self.checkpoint.items():
                    print("Loading {} model weight from {}".format(submodule, self.checkpoint[submodule]))
                    partially_load_checkpoint(self, submodule, checkpoint, map_location=self.device)
            else:
                print("Training model from scratch")
        except Exception:
            print("Error when trying to load checkpoint!")
