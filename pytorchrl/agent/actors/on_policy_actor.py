import gym
import numpy as np
import torch
import torch.nn as nn

from pytorchrl.agent.actors.distributions import get_dist
from pytorchrl.agent.actors.utils import Scale, Unscale, init
from pytorchrl.agent.actors.memory_networks import GruNet
from pytorchrl.agent.actors.feature_extractors import MLP, default_feature_extractor


class OnPolicyActor(nn.Module):
    """
    Actor critic class for On-Policy algorithms.

    It contains a policy network to predict next actions and a critic
    value network to predict the value score of a given obs.

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space.
    recurrent_nets : bool
        Whether to use a RNNs on top of the feature extractors.
    recurrent_nets_kwargs:
        Keyword arguments for the memory network.
    feature_extractor_network : nn.Module
        PyTorch nn.Module used as the features extraction block in all networks.
    feature_extractor_kwargs : dict
        Keyword arguments for the feature extractor network.
    shared_policy_value_network : bool
        Whether or not to share weights between policy and value networks.
    """
    def __init__(self,
                 input_space,
                 action_space,
                 recurrent_nets=False,
                 recurrent_nets_kwargs={},
                 feature_extractor_network=None,
                 feature_extractor_kwargs={},
                 shared_policy_value_network=True):

        """

        This actor defines policy network as:
        -------------------------------------

        policy = obs_feature_extractor + memory_net + action_distribution

        defines value network as:
        -------------------------

        value = obs_feature_extractor + memory_net + v_prediction_layer

        and defines shared policy-value network as:
        -------------------------------------------
                                                     action_distribution
        value = obs_feature_extractor + memory_net +
                                                     v_prediction_layer
        """

        super(OnPolicyActor, self).__init__()
        self.input_space = input_space
        self.action_space = action_space
        self.recurrent_nets = recurrent_nets
        self.shared_policy_value_network = shared_policy_value_network

        # If feature_extractor_network not defined, take default one based on input_space
        feature_extractor = feature_extractor_network or default_feature_extractor(input_space)

        #######################################################################
        #                           POLICY NETWORK                            #
        #######################################################################

        # ---- 1. Define obs feature extractor --------------------------------

        self.policy_feature_extractor = feature_extractor(
            input_space, **feature_extractor_kwargs)

        # ---- 2. Define memory network  --------------------------------------

        feature_size = int(np.prod(self.policy_feature_extractor(
            torch.randn(1, *input_space.shape)).shape))

        self.recurrent_size = feature_size
        if recurrent_nets:
            self.policy_memory_net = GruNet(feature_size, **recurrent_nets_kwargs)
            feature_size = self.policy_memory_net.num_outputs
        else:
            self.policy_memory_net = nn.Identity()

        # ---- 3. Define action distribution ----------------------------------

        if isinstance(action_space, gym.spaces.Discrete):
            self.dist = get_dist("Categorical")(feature_size, action_space.n)
            self.scale = None
            self.unscale = None

        elif isinstance(action_space, gym.spaces.Box):  # Continuous action space
            self.dist = get_dist("Gaussian")(feature_size, action_space.shape[0])
            self.scale = Scale(action_space)
            self.unscale = Unscale(action_space)

        elif isinstance(action_space, gym.spaces.Dict):
            raise NotImplementedError

        else:
            raise ValueError("Unrecognized action space")

        # ---- 4. Concatenate all policy modules ------------------------------

        self.policy_net = nn.Sequential(
            self.policy_feature_extractor,
            self.policy_memory_net, self.dist)

        #######################################################################
        #                           VALUE NETWORK                             #
        #######################################################################

        if self.shared_policy_value_network:
            self.value_feature_extractor = nn.Identity()
            self.value_memory_net = nn.Identity()
            self.last_action_features = None
            self.last_action_rhs = None

        else:

            # ---- 1. Define obs feature extractor ----------------------------

            self.value_feature_extractor = feature_extractor(
                input_space, **feature_extractor_kwargs)

            # ---- 2. Define memory network  ----------------------------------

            if recurrent_nets:
                self.value_memory_net = GruNet(self.recurrent_size, **recurrent_nets_kwargs)
            else:
                self.value_memory_net = nn.Identity()

        # ---- 3. Define value predictor --------------------------------------

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.value_predictor = init_(nn.Linear(self.recurrent_size, 1))

        # ---- 4. Concatenate all value net modules ---------------------------

        self.value_net = nn.Sequential(
            self.value_feature_extractor,
            self.value_memory_net, self.value_predictor)

    @classmethod
    def create_factory(
            cls,
            input_space,
            action_space,
            restart_model=None,
            recurrent_nets=False,
            feature_extractor_kwargs={},
            feature_extractor_network=None,
            shared_policy_value_network=True):
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
        recurrent_nets : bool
            Whether to use a RNNs as feature extractors.
        shared_policy_value_network : bool
            Whether or not to share weights between policy and value networks.

        Returns
        -------
        create_actor_critic_instance : func
            creates a new OnPolicyActor class instance.
        """

        def create_actor_critic_instance(device):
            """Create and return an actor critic instance."""
            policy = cls(input_space=input_space,
                         action_space=action_space,
                         recurrent_nets=recurrent_nets,
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
        """Returns True if the actor network are recurrent."""
        return self.recurrent_nets

    @property
    def recurrent_hidden_state_size(self):
        """Size of policy recurrent hidden state"""
        return self.policy_feature_extractor.recurrent_hidden_state_size

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
            Initial recurrent hidden states.
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
        rhs_act = torch.zeros(num_proc, self.recurrent_size).to(dev)

        rhs = {"rhs_act": rhs_act, "rhs_q1": rhs_act.clone(), "rhs_q2": rhs_act.clone()}

        return obs, rhs, done

    def get_action(self, obs, rhs, done, deterministic=False):
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
        logp_action : torch.tensor
            Log probability of `action` within the predicted action distribution.
        rhs : dict
            Updated recurrent hidden states.
        entropy_dist : torch.tensor
            Entropy of the predicted action distribution.
        """

        action_features = self.policy_feature_extractor(obs)
        if self.recurrent_nets:
            action_features, rhs["rhs_act"] = self.policy_memory_net(
                action_features, rhs["rhs_act"], done)
        (action, clipped_action, logp_action, entropy_dist) = self.dist(
            action_features, deterministic=deterministic)

        self.last_action_features = action_features
        self.last_action_rhs = rhs["rhs_act"]

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

        if self.scale:
            action = self.scale(action)

        features = self.policy_feature_extractor(obs)
        if self.recurrent_nets:
            action_features, rhs["rhs_act"] = self.policy_memory_net(
                features, rhs["rhs_act"], done)
        logp_action, entropy_dist = self.dist.evaluate_pred(features, action)

        self.last_action_features = features
        self.last_action_rhs = rhs["rhs_act"]

        return logp_action, entropy_dist, rhs

    def get_value(self, obs, rhs, done):
        """
        Return value scores of given observation.

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

        if self.shared_policy_value_network:
            if self.last_action_features.shape[0] != done.shape[0]:
                _, _, _, _, _ = self.get_action(obs, rhs["rhs_act"], done)
            return self.value_predictor(self.last_action_features), rhs

        else:
            value_features = self.value_feature_extractor(obs)
            if self.recurrent_nets:
                value_features, rhs["rhs_val"] = self.value_memory_net(
                    value_features, rhs["rhs_val"], done)
            return self.value_predictor(value_features), rhs
