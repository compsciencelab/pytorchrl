import torch
import torch.nn as nn

from .utils import Scale, Unscale
from .neural_networks import NNBase
from .distributions import get_dist
from .neural_networks.feature_extractors.utils import init
from .neural_networks.feature_extractors import get_feature_extractor


class OnPolicyActor(nn.Module):
    """
    Actor critic class for On-Policy algorithms.

    It contains a policy network (actor) to predict next actions and a critic
    value network to predict the value score of a given obs.

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space.
    feature_extractor_network : nn.Module
        PyTorch nn.Module used as the features extraction block in all networks.
    feature_extractor_kwargs : dict
        Keyword arguments for the feature extractor network.
    recurrent_policy : bool
        Whether to use a RNN as a policy.
    recurrent_hidden_size : int
        Policy RNN hidden size.
    shared_policy_value_network : bool
        Whether or not to share weights between policy and value networks.

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
        from the features extracted by the policy_net.
    value_net : nn.module
        Neural network that predicts a value score for a given env obs.
    shared_policy_value_network : bool
        If True, the feature extraction block of the value_net will be the
        policy_net.
    scale : nn.module
        Maps actions from [space.low, space.high] range to [-1, 1] range.
    unscale : nn.module
        Maps actions from [-1, 1] range to [space.low, space.high] range.

    """
    def __init__(self,
                 input_space,
                 action_space,
                 feature_extractor_network=get_feature_extractor("MLP"),
                 feature_extractor_kwargs={},
                 recurrent_policy=False,
                 recurrent_hidden_size=256,
                 shared_policy_value_network=True):

        super(OnPolicyActor, self).__init__()
        self.input_space = input_space
        self.action_space = action_space
        self.shared_policy_value_network = shared_policy_value_network

        # ---- Input/Output spaces --------------------------------------------

        policy_inputs = [input_space.shape]
        value_inputs = [input_space.shape]

        # ---- Networks -------------------------------------------------------

        self.policy_net = NNBase(
            policy_inputs, feature_extractor_network,
            feature_extractor_kwargs, recurrent=recurrent_policy,
            recurrent_hidden_size=recurrent_hidden_size, final_activation=True)

        if self.shared_policy_value_network:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
            self.value_net = init_(nn.Linear(self.policy_net.num_outputs, 1))
            self.last_actor_features = None

        else:
            self.value_net = NNBase(
                value_inputs, feature_extractor_network,
                feature_extractor_kwargs, output_shape=(1,))

        # ---- Distributions --------------------------------------------------

        if action_space.__class__.__name__ == "Discrete":
            self.dist = get_dist("Categorical")(self.policy_net.num_outputs, action_space.n)
            self.scale = None
            self.unscale = None

        elif action_space.__class__.__name__ == "Box":  # Continuous action space
            self.dist = get_dist("Gaussian")(self.policy_net.num_outputs, action_space.shape[0])
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
            recurrent_hidden_size=256,
            feature_extractor_kwargs={},
            shared_policy_value_network=True,
            feature_extractor_network=get_feature_extractor("MLP")):
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
        shared_policy_value_network : bool
            Whether or not to share weights between policy and value networks.

        Returns
        -------
        create_actor_critic_instance : func
            creates a new OnPolicyActorCritic class instance.
        """

        def create_actor_critic_instance(device):
            """Create and return an actor critic instance."""
            policy = cls(input_space=input_space,
                         action_space=action_space,
                         recurrent_policy=recurrent_policy,
                         recurrent_hidden_size=recurrent_hidden_size,
                         feature_extractor_kwargs=feature_extractor_kwargs,
                         feature_extractor_network=feature_extractor_network,
                         shared_policy_value_network=shared_policy_value_network)
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

    def get_action(self, obs, rhs, done, deterministic=False):
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

        actor_features, rhs = self.policy_net(obs, rhs, done)
        self.last_actor_features = actor_features

        (action, clipped_action, logp_action, entropy_dist) = self.dist(
            actor_features, deterministic=deterministic)

        if self.unscale:
            action = self.unscale(action)
            clipped_action = self.unscale(clipped_action)

        return action, clipped_action, logp_action, rhs, entropy_dist

    def evaluate_actions(self, obs, rhs, done, action):
        """
        Evaluate log likelihood of action given obs and the current
        policy network. Returns also entropy distribution.

        Parameters
        ----------
        obs : torch.tensor
            Environment observation.
        rhs : torch.tensor
            Recurrent hidden state.
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
        rhs : torch.tensor
            Updated recurrent hidden state.
        """

        if self.scale:
            action = self.scale(action)

        actor_features, rhs = self.policy_net(obs, rhs, done)
        logp_action, entropy_dist = self.dist.evaluate_pred(actor_features, action)
        self.last_actor_features = actor_features

        return logp_action, entropy_dist, rhs

    def get_value(self, obs, rhs, done):
        """
        Return value scores of given observation.

        Parameters
        ----------
        obs : torch.tensor
            Environment observation.

        Returns
        -------
        value : torch.tensor
            value score according to current value_net version.
        """
        if self.shared_policy_value_network:
            if self.last_actor_features.shape[0] != obs.shape[0]:
                self.last_actor_features, _ = self.policy_net(obs, rhs, done)
            return self.value_net(self.last_actor_features)
        else:
            val, _ = self.value_net(obs)
            return val

