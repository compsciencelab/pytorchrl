import gym
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

import pytorchrl as prl
from pytorchrl.agent.actors.base import Actor
from pytorchrl.agent.actors.distributions import get_dist
from pytorchrl.agent.actors.utils import Scale, Unscale, init
from pytorchrl.agent.actors.memory_networks import GruNet
from pytorchrl.agent.actors.feature_extractors import default_feature_extractor


class OnPolicyActor(Actor):
    """
    Actor critic class for On-Policy algorithms.

    It contains a policy network to predict next actions and a critic
    value network to predict the value score of a given obs.

    Parameters
    ----------
    device: torch.device
        CPU or specific GPU where class computations will take place.
    input_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space.
    algorithm_name : str
        Name of the RL algorithm used for learning.
    checkpoint : str
        Path to a previously trained Actor checkpoint to be loaded.
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
                 device,
                 input_space,
                 action_space,
                 algorithm_name,
                 checkpoint=None,
                 recurrent_nets=False,
                 recurrent_nets_kwargs={},
                 feature_extractor_network=None,
                 feature_extractor_kwargs={},
                 shared_policy_value_network=True):

        super(OnPolicyActor, self).__init__(
            device=device,
            checkpoint=checkpoint,
            input_space=input_space,
            action_space=action_space)

        self.recurrent_nets = recurrent_nets
        self.recurrent_nets_kwargs = recurrent_nets_kwargs
        self.feature_extractor_network = feature_extractor_network
        self.shared_policy_value_network = shared_policy_value_network
        self.feature_extractor_kwargs = feature_extractor_kwargs

        if algorithm_name in (prl.A2C, prl.PPO):
            self.num_critics_ext = 1
            self.num_critics_int = 0
        elif algorithm_name in (prl.RND_PPO):
            self.num_critics_ext = 1
            self.num_critics_int = 1

        # ----- Policy Network ----------------------------------------------------

        self.create_policy("policy_net")

        # ----- Value Networks ----------------------------------------------------

        for i in range(self.num_critics_ext):
            self.create_critic("value_net{}".format(i + 1))

        for i in range(self.num_critics_int):
            self.create_critic("ivalue_net{}".format(i + 1))

    @classmethod
    def create_factory(
            cls,
            input_space,
            action_space,
            algorithm_name,
            restart_model=None,
            recurrent_nets=False,
            recurrent_nets_kwargs={},
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
        algorithm_name : str
            Name of the RL algorithm_name used for learning.
        restart_model : str
            Path to a previously trained Actor checkpoint to be loaded.
        feature_extractor_network : nn.Module
            PyTorch nn.Module used as the features extraction block in all networks.
        feature_extractor_kwargs : dict
            Keyword arguments for the feature extractor network.
        recurrent_nets : bool
            Whether to use a RNNs as feature extractors.
        recurrent_nets_kwargs:
            Keyword arguments for the memory network.
        shared_policy_value_network : bool
            Whether or not to share weights between policy and value networks.

        Returns
        -------
        create_actor_critic_instance : func
            creates a new OnPolicyActor class instance.
        """

        def create_actor_critic_instance(device):
            """Create and return an actor critic instance."""
            policy = cls(device=device,
                         input_space=input_space,
                         action_space=action_space,
                         algorithm_name=algorithm_name,
                         recurrent_nets=recurrent_nets,
                         checkpoint=restart_model,
                         recurrent_nets_kwargs=recurrent_nets_kwargs,
                         feature_extractor_kwargs=feature_extractor_kwargs,
                         feature_extractor_network=feature_extractor_network,
                         shared_policy_value_network=shared_policy_value_network)
            policy.to(device)

            try:
                policy.try_load_from_checkpoint()
            except RuntimeError:
                pass

            return policy

        return create_actor_critic_instance

    @property
    def is_recurrent(self):
        """Returns True if the actor network are recurrent."""
        return self.recurrent_nets

    @property
    def recurrent_hidden_state_size(self):
        """Size of policy recurrent hidden state"""
        return self.recurrent_size

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
        rhs_policy = torch.zeros(num_proc, self.recurrent_size).to(dev)

        rhs = {"policy": rhs_policy}
        rhs.update({"value_net{}".format(i + 1): rhs_policy.clone() for i in range(self.num_critics_ext)})
        rhs.update({"ivalue_net{}".format(i + 1): rhs_policy.clone() for i in range(self.num_critics_int)})

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
        dist : torch.Distribution
            Predicted probability distribution over next action.
        """

        features = self.policy_net.feature_extractor(obs)
        if self.recurrent_nets:
            features, rhs["policy"] = self.policy_net.memory_net(
                features, rhs["policy"], done)
        (action, clipped_action, logp_action, entropy_dist, dist) = self.policy_net.dist(
            features, deterministic=deterministic)

        self.last_action_features = features

        if self.unscale:
            action = self.unscale(action)
            clipped_action = self.unscale(clipped_action)

        return action, clipped_action, logp_action, rhs, entropy_dist, dist

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
        dist : torch.Distribution
            Predicted probability distribution over next action.
        """

        if self.scale:
            action = self.scale(action)

        features = self.policy_net.feature_extractor(obs)

        if self.recurrent_nets:
            features, rhs["policy"] = self.policy_net.memory_net(
                features, rhs["policy"], done)

        logp_action, entropy_dist, dist = self.policy_net.dist.evaluate_pred(features, action)

        self.last_action_features = features

        return logp_action, entropy_dist, dist

    def get_value_specific_net(self, obs, rhs, done, value_net_name):
        """
        Return value score for a single value network.

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
            Predicted value score.
        rhs : dict
            Updated recurrent hidden states.
        """

        value_net = getattr(self, value_net_name)

        if self.shared_policy_value_network:
            if self.last_action_features.shape[0] != done.shape[0]:
                _, _, _, _, _, _ = self.get_action(obs, rhs["policy"], done)
            value = value_net.predictor(self.last_action_features)

        else:
            value_features = value_net.feature_extractor(obs)
            if self.recurrent_nets:
                value_features, rhs[value_net_name] = value_net.memory_net(
                    value_features, rhs[value_net_name], done)
            value = value_net.predictor(value_features)

        return value, rhs

    def get_value(self, obs, rhs, done):
        """
        Return all value scores of given observation.

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
        output : dict
            Dict containing value prediction from each critic under keys "value_net1", "value_net2", etc
            as well as the recurrent hidden states under the key "rhs".
        """

        outputs = {}

        for i in range(self.num_critics_ext):
            value_net_name = "value_net{}".format(i + 1)
            value, rhs = self.get_value_specific_net(obs, rhs, done, value_net_name)
            outputs[value_net_name] = value

        for i in range(self.num_critics_int):
            value_net_name = "ivalue_net{}".format(i + 1)
            value, rhs = self.get_value_specific_net(obs, rhs, done, value_net_name)
            outputs[value_net_name] = value

        outputs["rhs"] = rhs
        return outputs

    def create_critic(self, name):
        """
        Create a critic value network and define it as class attribute under the name `name`.
        This actor defines defines value networks as:

        value = obs_feature_extractor + memory_net + v_prediction_layer

        and defines shared policy-value network as:
                                                     action_distribution
        value = obs_feature_extractor + memory_net +
                                                     v_prediction_layer

        Parameters
        ----------
        name : str
            Critic network name.
        """

        # If feature_extractor_network not defined, take default one based on input_space
        feature_extractor = self.feature_extractor_network or default_feature_extractor(self.input_space)

        if self.shared_policy_value_network:
            value_feature_extractor = nn.Identity()
            value_memory_net = nn.Identity()
            self.last_action_features = None

        else:

            # ---- 1. Define obs feature extractor ----------------------------

            value_feature_extractor = feature_extractor(
                self.input_space, **self.feature_extractor_kwargs)

            # ---- 2. Define memory network  ----------------------------------

            feature_size = int(np.prod(value_feature_extractor(
                torch.randn(1, *self.input_space.shape)).shape))

            if self.recurrent_nets:
                value_memory_net = GruNet(feature_size, **self.recurrent_nets_kwargs)
            else:
                value_memory_net = nn.Identity()

        # ---- 3. Define value predictor --------------------------------------

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(0.01))
        value_predictor = init_(nn.Linear(self.recurrent_size, 1))

        # ---- 4. Concatenate all value net modules ---------------------------

        v_net = nn.Sequential(OrderedDict([
            ('feature_extractor', value_feature_extractor),
            ('memory_net', value_memory_net),
            ("predictor", value_predictor),
        ]))

        setattr(self, name, v_net)

    def create_policy(self, name):
        """
        Create a policy network and define it as class attribute under the name `name`.
        This actor defines policy network as:

        policy = obs_feature_extractor + memory_net + action_distribution

        Parameters
        ----------
        name : str
            Policy network name.
        """

        # If feature_extractor_network not defined, take default one based on input_space
        feature_extractor = self.feature_extractor_network or default_feature_extractor(self.input_space)

        # ---- 1. Define obs feature extractor --------------------------------

        policy_feature_extractor = feature_extractor(
            self.input_space, **self.feature_extractor_kwargs)

        # ---- 2. Define memory network  --------------------------------------

        feature_size = int(np.prod(policy_feature_extractor(
            torch.randn(1, *self.input_space.shape)).shape))

        if self.recurrent_nets:
            policy_memory_net = GruNet(feature_size, **self.recurrent_nets_kwargs)
            self.recurrent_size = policy_memory_net.recurrent_hidden_state_size
        else:
            policy_memory_net = nn.Identity()
            self.recurrent_size = feature_size

        # ---- 3. Define action distribution ----------------------------------

        if isinstance(self.action_space, gym.spaces.Discrete):
            dist = get_dist("Categorical")(self.recurrent_size, self.action_space.n)
            self.scale = None
            self.unscale = None

        elif isinstance(self.action_space, gym.spaces.Box):  # Continuous action space
            dist = get_dist("Gaussian")(self.recurrent_size, self.action_space.shape[0])
            self.scale = Scale(self.action_space)
            self.unscale = Unscale(self.action_space)

        elif isinstance(self.action_space, gym.spaces.Dict):
            raise NotImplementedError

        else:
            raise ValueError("Unrecognized action space")

        # ---- 4. Concatenate all policy modules ------------------------------

        policy_net = nn.Sequential(OrderedDict([
            ('feature_extractor', policy_feature_extractor),
            ('memory_net', policy_memory_net),
            ('dist', dist),
        ]))

        setattr(self, name, policy_net)
