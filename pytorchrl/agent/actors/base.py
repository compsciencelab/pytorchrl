import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Actor(nn.Module, ABC):
    """
    Base class for all Actors.

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space.
    """
    def __init__(self,
                 input_space,
                 action_space,
                 *args):

        super(Actor, self).__init__()
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_factory(
            cls,
            input_space,
            action_space,
            *args):
        """
        Returns a function that creates actor critic instances.

        Parameters
        ----------
        input_space : gym.Space
            Environment observation space.
        action_space : gym.Space
            Environment action space.

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

    @abstractmethod
    def burn_in_recurrent_states(self, data_batch):
        """
        Applies a recurrent burn-in phase to data_batch as described in
        (https://openreview.net/pdf?id=r1lyTjAqYX). First T steps are used
        to compute on-policy recurrent hidden states. data_batch is then
        updated, discarding T first steps in all tensors.

        Parameters
        ----------
        data_batch : dict
            data batch containing all required tensors to compute Algorithm loss.

        Returns
        -------
        data_batch : dict
            Updated data batch after burn-in phase.
        """
        raise NotImplementedError

    @abstractmethod
    def get_action(self, obs, rhs, done, deterministic=False, *args):
        """
        Predict and return next action, along with other information.

        Parameters
        ----------
        obs : torch.tensor
            Current environment observation.
        rhs : dict
            Current recurrent hidden states.
        done : torch.tensor
            Current done tensor, indicating if episode has finished.
        deterministic : bool
            Whether to randomly sample action from predicted distribution or take the mode.

        Returns
        -------
        action : torch.tensor
            Next action sampled.
        clipped_action : torch.tensor
            Next action sampled, but clipped to be within the env action space.
        rhs : dict
            Updated recurrent hidden states.
        other : torch.tensor
            Additional tensors from specific Actor.
        """
        raise NotImplementedError

    def evaluate_actions(self, obs, rhs, done, action):
        """
        Evaluate log likelihood of action given obs and the current
        policy network. Returns also entropy distribution.

        Parameters
        ----------
        obs : torch.tensor
            Environment observation.
        rhs : dict
            Recurrent hidden states.
        done : torch.tensor
            Done tensor, indicating if episode has finished.
        action : torch.tensor
            Evaluated action.

        Returns
        -------
        logp_action : torch.tensor
            Log probability of `action` according to the action distribution
            predicted with current version of the policy_net.
        entropy_dist : torch.tensor
            Entropy of the action distribution predicted with current version
            of the policy_net.
        rhs : dict
            Updated recurrent hidden states.
        """
        raise NotImplementedError

    def get_q_scores(self, obs, rhs, done, actions=None):
        """
        If actor has Q-networks, return Q scores of the given observations
        and actions.

        Parameters
        ----------
        obs : torch.tensor
            Environment observation.
        rhs : dict
            Current recurrent hidden states.
        done : torch.tensor
            Current done tensor, indicating if episode has finished.
        actions : torch.tensor
             Evaluated actions.

        Returns
        -------
        q1 : torch.tensor
            Q score according to current q1 network version.
        q2 : torch.tensor
            Q score according to current q2 network version. If there is no
            q2 network, return None
        rhs : dict
            Updated recurrent hidden states.
        """
        raise NotImplementedError

    def get_value(self, obs, rhs, done):
        """
        If actor has a value network, return value scores of given observation.

        Parameters
        ----------
        obs : torch.tensor
            Environment observation.
        rhs : dict
            Recurrent hidden states.
        done : torch.tensor
            Done tensor, indicating if episode has finished.

        Returns
        -------
        value : torch.tensor
            value score according to current value_net version.
        rhs : dict
            Updated recurrent hidden states.
        """
        raise NotImplementedError
