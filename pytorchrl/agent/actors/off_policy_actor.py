import torch
import torch.nn as nn

from .utils import Scale, Unscale
from .neural_networks import NNBase
from .distributions import get_dist
from .neural_networks.feature_extractors import get_feature_extractor


class OffPolicyActor(nn.Module):
    """
    Actor critic class for Off-Policy algorithms.

    It contains a policy network (actor) to predict next actions and one or two
    Q networks.

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space.
    feature_extractor_network : nn.Module
        PyTorch nn.Module to extract features in all networks.
    feature_extractor_kwargs : dict
        Keyword arguments for the feature extractor network.
    recurrent_policy : bool
        Whether to use a RNN as a policy.
    recurrent_hidden_size : int
        Policy RNN hidden size.
    create_double_q_critic : bool
        Whether to instantiate a second Q network or not.

    Attributes
    ----------
    policy_net : nn.module
        Neural network that extracts features from the input env obs.
    input_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space.
    dist : nn.module
        Neural network that predict a prob distribution over the action space
    q1 : nn.module
        Neural network that predicts a Q-value for a given env obs and action.
    q2 : nn.module
        A second neural network to predict a Q-value for a given env obs and action.
    scale : nn.module
        Maps actions from [space.low, space.high] range to [-1, 1] range.
    unscale : nn.module
        Maps actions from [-1, 1] range to [space.low, space.high] range.

    Examples
    --------
    """
    def __init__(self,
                 input_space,
                 action_space,
                 feature_extractor_network=get_feature_extractor("MLP"),
                 feature_extractor_kwargs={},
                 recurrent_policy=False,
                 recurrent_hidden_size=512,
                 create_double_q_critic=True):

        super(OffPolicyActor, self).__init__()
        self.input_space = input_space
        self.action_space = action_space

        # ---- Input/Output spaces --------------------------------------------

        policy_inputs = [input_space.shape]

        if action_space.__class__.__name__ == "Discrete":
            q_inputs = [input_space.shape]
            q_outputs = (action_space.n,)

        elif action_space.__class__.__name__ == "Box":
            q_outputs = (1,)
            if len(input_space.shape) == 3: # observations are images
                q_inputs = [input_space.shape, action_space.shape]
            elif len(input_space.shape) == 1:
                q_inputs = [(input_space.shape[0] + action_space.shape[0],)]
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        # ---- Networks -------------------------------------------------------

        self.policy_net = NNBase(
            policy_inputs, feature_extractor_network,
            feature_extractor_kwargs, recurrent=recurrent_policy,
            recurrent_hidden_size=recurrent_hidden_size, final_activation=True)

        self.q1 = NNBase(
            q_inputs, feature_extractor_network,
            feature_extractor_kwargs, output_shape=q_outputs)

        if create_double_q_critic:
            self.q2 = NNBase(
                q_inputs, feature_extractor_network,
                feature_extractor_kwargs, output_shape=q_outputs)
        else:
            self.q2 = None

        # ---- Distributions --------------------------------------------------

        if action_space.__class__.__name__ == "Discrete":
            self.dist = get_dist("Categorical")(self.policy_net.num_outputs, action_space.n)
            self.scale = None
            self.unscale = None

        elif action_space.__class__.__name__ == "Box":  # Continuous action space
            self.dist = get_dist("SquashedGaussian")(self.policy_net.num_outputs, action_space.shape[0])
            self.scale = Scale(action_space)
            self.unscale = Unscale(action_space)

        else:
            raise NotImplementedError

    @classmethod
    def create_factory(
            cls,
            input_space,
            action_space,
            restart_model=None,
            recurrent_policy=False,
            recurrent_hidden_size=512,
            feature_extractor_kwargs={},
            feature_extractor_network=get_feature_extractor("MLP"),
            create_double_q_critic=True):
        """
        Returns a function that creates actor critic instances.

        Parameters
        ----------
        input_space : gym.Space
            Environment observation space.
        action_space : gym.Space
            Environment action space.
        restart_model : str
            Path to a previously trained ActorCritic checkpoint to be loaded.
        feature_extractor_network : nn.Module
            PyTorch nn.Module used as the features extraction block in all networks.
        feature_extractor_kwargs : dict
            Keyword arguments for the feature extractor network.
        recurrent_policy : bool
            Whether to use a RNN as a policy.
        recurrent_hidden_size : int
            Policy RNN hidden size.
        create_double_q_critic : bool
            whether to instantiate a second Q network or not.

        Returns
        -------
        create_actor_critic_instance : func
            creates a new OffPolicyActorCritic class instance.
        """

        def create_actor_critic_instance(device):
            """Create and return an actor critic instance."""
            policy = cls(input_space=input_space,
                         action_space=action_space,
                         recurrent_policy=recurrent_policy,
                         recurrent_hidden_size=recurrent_hidden_size,
                         create_double_q_critic=create_double_q_critic,
                         feature_extractor_kwargs=feature_extractor_kwargs,
                         feature_extractor_network=feature_extractor_network)
            if restart_model:
                policy.load_state_dict(
                    torch.load(restart_model, map_location=device))
            policy.to(device)
            return policy

        return create_actor_critic_instance

    @property
    def is_recurrent(self):
        """Returns True if the policy network has recurrency."""
        return self.policy_net.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of policy recurrent hidden state"""
        return self.policy_net.recurrent_hidden_state_size

    def policy_initial_states(self, obs):
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
            Initial recurrent hidden state (will contain zeroes).
        done : torch.tensor
            Initial done tensor, indicating the environment is not done.
        """
        obs, rhs, done = self.policy_net.initial_states(obs)
        return obs, rhs, done

    def get_action(self, obs, rhs, dones, deterministic=False):
        """
        Predict and return next action, along with other information.

        Parameters
        ----------
        obs : torch.tensor
            Current environment observation.
        rhs : torch.tensor
            Current recurrent hidden state.
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
        logp_action : torch.tensor
            Log probability of `action` within the predicted action distribution.
        rhs : torch.tensor
            Updated recurrent hidden state.
        entropy_dist : torch.tensor
            Entropy of the predicted action distribution.
        """
        actor_features, rhs = self.policy_net(obs, rhs, dones)
        self.last_actor_features = actor_features

        (action, clipped_action, logp_action, entropy_dist) = self.dist(
            actor_features, deterministic=deterministic)

        if self.unscale:
            action = self.unscale(action)
            clipped_action = self.unscale(clipped_action)

        return action, clipped_action, logp_action, rhs, entropy_dist

    def get_q_scores(self, obs, actions=None):
        """
        Return Q scores of the given observations and actions.
        Parameters
        ----------
        obs : torch.tensor
            Environment observation.
        actions : torch.tensor
             Evaluated actions.

        Returns
        -------
        q1 : torch.tensor
            Q score according to current q1 network version.
        q2 : torch.tensor
            Q score according to current q2 network version.
        """

        if actions is not None:
            if len(obs.shape[1:]) == 3: # obs are images
                inputs = (obs, actions)
            elif len(obs.shape[1:]) == 1:
                inputs = torch.cat([obs, actions], dim=-1)
        else:
            inputs = obs

        q1, _ = self.q1(inputs)
        q2, _ = self.q2(inputs) if self.q2 else None
        return q1, q2