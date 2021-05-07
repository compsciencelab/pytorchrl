from abc import ABC, abstractmethod
import torch


class Algorithm(ABC):
    """Base class for all algorithms"""

    @classmethod
    @abstractmethod
    def create_factory(cls):
        """Returns a function to create new Algo instances"""
        raise NotImplementedError

    @property
    @abstractmethod
    def gamma(self):
        """Returns discount factor gamma."""
        raise NotImplementedError

    @property
    @abstractmethod
    def start_steps(self):
        """Returns the number of steps to collect with initial random policy."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_epochs(self):
        """
        Returns the number of times the whole buffer is re-used before data
        collection proceeds.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def update_every(self):
        """
        Returns the number of data samples collected between
        network update stages.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_mini_batch(self):
        """
        Returns the number of times the whole buffer is re-used before data
        collection proceeds.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def mini_batch_size(self):
        """
        Returns the number of mini batches per epoch.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def test_every(self):
        """Number of network updates between test evaluations."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_test_episodes(self):
        """
        Returns the number of episodes to complete when testing.
        """
        raise NotImplementedError

    @abstractmethod
    def acting_step(self, obs, rhs, done, deterministic=False, *args):
        """
        Algorithm acting function.

        Parameters
        ----------
        obs: torch.tensor
            Current world observation
        rhs: torch.tensor
            RNN recurrent hidden state (if policy is not a RNN, rhs will contain zeroes).
        done: torch.tensor
            1.0 if current obs is the last one in the episode, else 0.0.
        deterministic: bool
            Whether to randomly sample action from predicted distribution or take the mode.

        Returns
        -------
        action: torch.tensor
            Predicted next action.
        clipped_action: torch.tensor
            Predicted next action (clipped to be within action space).
        rhs: torch.tensor
            Policy recurrent hidden state (if policy is not a RNN, rhs will contain zeroes).
        other: dict
            Additional PPO predictions, value score and action log probability,
            which are not used in other algorithms.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_gradients(self, batch, grads_to_cpu=True, *args):
        """
        Compute loss and compute gradients but don't do optimization step,
        return gradients instead.

        Parameters
        ----------
        data: dict
            data batch containing all required tensors to compute PPO loss.
        grads_to_cpu: bool
            If gradient tensor will be sent to another node, need to be in CPU.

        Returns
        -------
        grads: list of tensors
            List of actor_critic gradients.
        info: dict
            Dict containing current PPO iteration information.
        """

        raise NotImplementedError

    @abstractmethod
    def apply_gradients(self, gradients=None, *args):
        """
        Take an optimization step, previously setting new gradients if provided.

        Parameters
        ----------
        gradients: list of tensors
            List of actor_critic gradients.
        """
        raise NotImplementedError

    @abstractmethod
    def set_weights(self, actor_weights, *args):
        """
        Update actor with the given weights

        Parameters
        ----------
        actor_weights: dict of tensors
            Dict containing actor_critic weights to be set.
        """
        raise NotImplementedError

    @abstractmethod
    def update_algorithm_parameter(self, parameter_name, new_parameter_value, *args):
        """
        If `parameter_name` is an attribute of the algorithm, change its value
        to `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Attribute name
        new_parameter_value : int or float
            New value for `parameter_name`.
        """
        raise NotImplementedError
