import gym
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

import pytorchrl as prl
from pytorchrl.agent.actors.base import Actor
from pytorchrl.agent.actors.distributions import get_dist
from pytorchrl.agent.actors.utils import Scale, Unscale, init
from pytorchrl.agent.actors.memory_networks import GruNet
from pytorchrl.agent.actors.feature_extractors import MLP, default_feature_extractor, get_feature_extractor


class OffPolicyActor(Actor):
    """
    Actor critic class for Off-Policy algorithms.

    It contains a policy network (actor) to predict next actions and one or two
    Q networks.

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
    noise : str
        Type of exploration noise that will be added to the deterministic actions.
    obs_feature_extractor : nn.Module
        PyTorch nn.Module to extract features from observation in all networks.
    obs_feature_extractor_kwargs : dict
        Keyword arguments for the obs extractor network.
    act_feature_extractor : nn.Module
        PyTorch nn.Module to extract features from actions in all networks.
    act_feature_extractor_kwargs : dict
        Keyword arguments for the act extractor network.
    common_feature_extractor : nn.Module
        PyTorch nn.Module to extract joint features from the concatenation of
        action and observation features.
    common_feature_extractor_kwargs : dict
        Keyword arguments for the common extractor network.
    recurrent_nets : bool
        Whether to use a RNNs as feature extractors.
    sequence_overlap : float
        From 0.0 to 1.0, how much consecutive rollout sequences will overlap.
    recurrent_nets_kwargs : dict
        Keyword arguments for the memory network.
    num_critics : int
        Number of Q networks to be instantiated.

    Examples
    --------
    """
    def __init__(self,
                 device,
                 input_space,
                 action_space,
                 algorithm_name,
                 noise=None,
                 checkpoint=None,
                 sequence_overlap=0.5,
                 recurrent_nets=False,
                 recurrent_nets_kwargs={},
                 obs_feature_extractor=None,
                 obs_feature_extractor_kwargs={},
                 act_feature_extractor=None,
                 act_feature_extractor_kwargs={},
                 common_feature_extractor=MLP,
                 common_feature_extractor_kwargs={},
                 num_critics=2):

        super(OffPolicyActor, self).__init__(
            device=device,
            checkpoint=checkpoint,
            input_space=input_space,
            action_space=action_space)

        self.noise = noise
        self.algorithm_name = algorithm_name
        self.input_space = input_space
        self.action_space = action_space
        self.act_feature_extractor = act_feature_extractor
        self.act_feature_extractor_kwargs = act_feature_extractor_kwargs
        self.obs_feature_extractor = obs_feature_extractor
        self.obs_feature_extractor_kwargs = obs_feature_extractor_kwargs
        self.common_feature_extractor = common_feature_extractor
        self.common_feature_extractor_kwargs = common_feature_extractor_kwargs
        self.recurrent_nets = recurrent_nets
        self.recurrent_nets_kwargs = recurrent_nets_kwargs
        self.sequence_overlap = np.clip(sequence_overlap, 0.0, 1.0)
        self.num_critics = num_critics
        self.deterministic = algorithm_name in [prl.DDPG, prl.TD3]

        # ----- Policy Network ----------------------------------------------------

        self.create_policy("policy_net")

        # ----- Q Networks ----------------------------------------------------

        for i in range(num_critics):
            self.create_critic("q{}".format(i + 1))

    @classmethod
    def create_factory(
            cls,
            input_space,
            action_space,
            algorithm_name,
            noise=None,
            restart_model=None,
            sequence_overlap=0.5,
            recurrent_nets_kwargs={},
            recurrent_nets=False,
            obs_feature_extractor=None,
            obs_feature_extractor_kwargs={},
            act_feature_extractor=None,
            act_feature_extractor_kwargs={},
            common_feature_extractor=MLP,
            common_feature_extractor_kwargs={},
            num_critics=2

    ):
        """
        Returns a function that creates actor critic instances.

        Parameters
        ----------
        input_space : gym.Space
            Environment observation space.
        action_space : gym.Space
            Environment action space.
        algorithm_name : str
            Name of the RL algorithm used for learning.
        noise : str
            Type of exploration noise that will be added to the deterministic actions.
        obs_feature_extractor : nn.Module
            PyTorch nn.Module to extract features from observation in all networks.
        obs_feature_extractor_kwargs : dict
            Keyword arguments for the obs extractor network.
        act_feature_extractor : nn.Module
            PyTorch nn.Module to extract features from actions in all networks.
        act_feature_extractor_kwargs : dict
            Keyword arguments for the act extractor network.
        common_feature_extractor : nn.Module
            PyTorch nn.Module to extract joint features from the concatenation of
            action and observation features.
        common_feature_extractor_kwargs : dict
            Keyword arguments for the common extractor network.
        recurrent_nets : bool
            Whether to use a RNNs as feature extractors.
        sequence_overlap : float
            From 0.0 to 1.0, how much consecutive rollout sequences will overlap.
        recurrent_nets_kwargs : dict
            Keyword arguments for the memory network.
        num_critics : int
            Number of Q networks to be instantiated.
        restart_model : str
            Path to a previously trained Actor checkpoint to be loaded.

        Returns
        -------
        create_actor_instance : func
            creates a new OffPolicyActor class instance.
        """

        def create_actor_instance(device):
            """Create and return an actor critic instance."""
            policy = cls(noise=noise,
                         device=device,
                         input_space=input_space,
                         action_space=action_space,
                         algorithm_name=algorithm_name,
                         checkpoint=restart_model,
                         sequence_overlap=sequence_overlap,
                         recurrent_nets_kwargs=recurrent_nets_kwargs,
                         recurrent_nets=recurrent_nets,
                         obs_feature_extractor=obs_feature_extractor,
                         obs_feature_extractor_kwargs=obs_feature_extractor_kwargs,
                         act_feature_extractor=act_feature_extractor,
                         act_feature_extractor_kwargs=act_feature_extractor_kwargs,
                         common_feature_extractor=common_feature_extractor,
                         common_feature_extractor_kwargs=common_feature_extractor_kwargs,
                         num_critics=num_critics)
            policy.to(device)

            try:
                policy.try_load_from_checkpoint()
            except RuntimeError:
                pass

            return policy

        return create_actor_instance

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
        rhs_act = torch.zeros(num_proc, self.recurrent_size).to(dev)

        rhs = {"rhs_act": rhs_act}
        rhs.update({"rhs_q{}".format(i + 1): rhs_act.clone() for i in range(self.num_critics)})

        return obs, rhs, done

    def burn_in_recurrent_states(self, data_batch):
        """
        Applies a recurrent burn-in phase to data_batch as described in
        (https://openreview.net/pdf?id=r1lyTjAqYX). Initial B steps are used
        to compute on-policy recurrent hidden states. data_batch is then
        updated, discarding B first steps in all tensors.

        Parameters
        ----------
        data_batch : dict
            data batch containing all required tensors to compute Algorithm loss.

        Returns
        -------
        data_batch : dict
            Updated data batch after burn-in phase.
        """

        # (T, N, -1) tensors that have been flatten to (T * N, -1)
        N = data_batch[prl.RHS]["rhs_act"].shape[0]  # number of sequences
        T = int(data_batch[prl.DONE].shape[0] / N)  # sequence lengths
        B = int(self.sequence_overlap * T)  # sequence burn-in length

        if B == 0:
            return data_batch

        # Split tensors into burn-in and no-burn-in
        chunk_sizes = [B, T - B] * N
        burn_in_data = {k: {} for k in data_batch}
        non_burn_in_data = {k: {} for k in data_batch}
        for k, v in data_batch.items():

            if k in (prl.RHS, prl.RHS2):
                burn_in_data[k] = v
                continue
            if not isinstance(v, (torch.Tensor, dict)):
                non_burn_in_data[k] = v
                continue
            if isinstance(v, dict):
                for x, y in v.items():
                    if not isinstance(y, torch.Tensor):
                        non_burn_in_data[k][x] = v
                        continue
                    sequence_slices = torch.split(y, chunk_sizes)
                    burn_in_data[k][x] = torch.cat(sequence_slices[0::2])
                    non_burn_in_data[k][x] = torch.cat(sequence_slices[1::2])
            else:
                sequence_slices = torch.split(v, chunk_sizes)
                burn_in_data[k] = torch.cat(sequence_slices[0::2])
                non_burn_in_data[k] = torch.cat(sequence_slices[1::2])

        # Do burn-in
        with torch.no_grad():

            act, _, _, rhs, _, _ = self.get_action(
                burn_in_data[prl.OBS], burn_in_data[prl.RHS], burn_in_data[prl.DONE])
            act2, _, _, rhs2, _, _ = self.get_action(
                burn_in_data[prl.OBS2], burn_in_data[prl.RHS2], burn_in_data[prl.DONE2])

            rhs = self.get_q_scores(
                burn_in_data[prl.OBS], rhs, burn_in_data[prl.DONE], act).get("rhs")
            rhs2 = self.get_q_scores(
                burn_in_data[prl.OBS2], rhs2, burn_in_data[prl.DONE2], act2).get("rhs")

            for k in rhs:
                rhs[k] = rhs[k].detach()
            for k in rhs2:
                rhs2[k] = rhs2[k].detach()

            non_burn_in_data[prl.RHS] = rhs
            non_burn_in_data[prl.RHS2] = rhs2

        return non_burn_in_data

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

        x = self.policy_net.common_feature_extractor(self.policy_net.obs_feature_extractor(obs))

        if self.recurrent_nets:
            x, rhs["rhs_act"] = self.policy_net.memory_net(x, rhs["rhs_act"], done)

        (action, clipped_action, logp_action, entropy_dist, dist) = self.policy_net.dist(
            x, deterministic=deterministic)

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

        features = self.policy_net.common_feature_extractor(self.policy_obs_feature_extractor(obs))

        if self.recurrent_nets:
            features, rhs["rhs_act"] = self.policy_net.memory_net(features, rhs["rhs_act"], done)

        logp_action, entropy_dist, dist = self.policy_net.dist.evaluate_pred(features, action)

        return logp_action, entropy_dist, dist

    def get_q_scores(self, obs, rhs, done, actions=None):
        """
        Return Q scores of the given observations and actions.

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
        output : dict
            Dict containing value prediction from each critic under keys "q1", "q2", etc
            as well as the recurrent hidden states under the key "rhs".
        """

        outputs = {}
        for i in range(self.num_critics):
            q = getattr(self, "q{}".format(i + 1))
            features = q.obs_feature_extractor(obs)

            if actions is not None:
                act_features = q.act_feature_extractor(actions)
                features = torch.cat([features, act_features], -1)
            features = q.common_feature_extractor(features)

            if self.recurrent_nets:
                features, rhs["rhs_q{}".format(1 + 1)] = q.memory_net(
                    features, rhs["rhs_q{}".format(i + 1)], done)

            q_scores = q.predictor(features)
            outputs["q{}".format(i + 1)] = q_scores

        outputs["rhs"] = rhs
        return outputs

    def create_critic(self, name):
        """
        Create a critic q network and define it as class attribute under the name `name`.
        This actor defines defines q networks as:

            obs_feature_extractor
        q =                       + common_feature_extractor + memory_net + q_prediction_layer
            act_feature_extractor

        Parameters
        ----------
        name : str
            Critic network name.
        """

        # ---- 1. Define action feature extractor -----------------------------

        act_extractor = self.act_feature_extractor or nn.Identity
        q_act_feature_extractor = act_extractor(
            self.action_space, **self.act_feature_extractor_kwargs)

        # ---- 2. Define obs feature extractor -----------------------------

        obs_extractor = self.obs_feature_extractor or nn.Identity
        q_obs_feature_extractor = obs_extractor(
            self.input_space, **self.obs_feature_extractor_kwargs)
        obs_feature_size = int(np.prod(q_obs_feature_extractor(
            torch.randn(1, *self.input_space.shape)).shape))

        # ---- 3. Define shared feature extractor -----------------------------

        if isinstance(self.action_space, gym.spaces.Discrete):
            act_feature_size = 0
            q_outputs = self.action_space.n

        elif isinstance(self.action_space, gym.spaces.Box):
            act_feature_size = int(np.prod(q_act_feature_extractor(
                torch.randn(1, *self.action_space.shape)).shape)) if self.act_feature_extractor \
                else np.prod(self.action_space.shape)
            q_outputs = 1

        else:
            raise NotImplementedError

        feature_size = obs_feature_size + act_feature_size
        q_common_feature_extractor = self.common_feature_extractor(
            feature_size, **self.common_feature_extractor_kwargs)

        # ---- 4. Define memory network ---------------------------------------

        feature_size = int(np.prod(q_common_feature_extractor(
            torch.randn(1, feature_size)).shape))
        q_memory_net = GruNet(feature_size, **self.recurrent_nets_kwargs) if\
            self.recurrent_nets else nn.Identity()
        feature_size = q_memory_net.recurrent_hidden_state_size if self.recurrent_nets\
            else feature_size

        # ---- 5. Define prediction layer -------------------------------------

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        q_predictor = init_(nn.Linear(feature_size, q_outputs))

        # ---- 6. Concatenate all q1 net modules ------------------------------

        q_net = nn.Sequential(OrderedDict([
            ('obs_feature_extractor', q_obs_feature_extractor),
            ('act_feature_extractor', q_act_feature_extractor),
            ('common_feature_extractor', q_common_feature_extractor),
            ('memory_net', q_memory_net),
            ("predictor", q_predictor),
        ]))

        setattr(self, name, q_net)

    def create_policy(self, name):
        """
        Create a policy network and define it as class attribute under the name `name`.
        This actor defines policy network as:

        policy = obs_feature_extractor + common_feature_extractor + memory_net + action distribution

        Parameters
        ----------
        name : str
            Policy network name.
        """

        # ---- 1. Define Obs feature extractor --------------------------------

        if self.obs_feature_extractor:
            self.obs_feature_extractor = get_feature_extractor(self.obs_feature_extractor)
        else:
            self.obs_feature_extractor = default_feature_extractor(self.input_space)

        obs_extractor = self.obs_feature_extractor or nn.Identity
        policy_obs_feature_extractor = obs_extractor(
            self.input_space, **self.obs_feature_extractor_kwargs)

        # ---- 2. Define Common feature extractor -----------------------------

        feature_size = int(np.prod(policy_obs_feature_extractor(
            torch.randn(1, *self.input_space.shape)).shape))

        policy_common_feature_extractor = self.common_feature_extractor(
            feature_size, **self.common_feature_extractor_kwargs)

        # ---- 3. Define memory network  --------------------------------------

        feature_size = int(np.prod(policy_common_feature_extractor(
            torch.randn(1, feature_size)).shape))

        if self.recurrent_nets:
            policy_memory_net = GruNet(feature_size, **self.recurrent_nets_kwargs)
            self.recurrent_size = policy_memory_net.recurrent_hidden_state_size
        else:
            policy_memory_net = nn.Identity()
            self.recurrent_size = feature_size

        # ---- 4. Define action distribution ----------------------------------

        if isinstance(self.action_space, gym.spaces.Discrete):
            dist = get_dist("Categorical")(self.recurrent_size, self.action_space.n)
            self.scale = None
            self.unscale = None

        elif isinstance(self.action_space, gym.spaces.Box) and not self.deterministic:
            if self.algorithm_name in [prl.SAC]:
                dist = get_dist("SquashedGaussian")(self.recurrent_size, self.action_space.shape[0])
            else:
                dist = get_dist("Gaussian")(self.recurrent_size, self.action_space.shape[0])
            self.scale = Scale(self.action_space)
            self.unscale = Unscale(self.action_space)

        elif isinstance(self.action_space, gym.spaces.Box) and self.deterministic:
            dist = get_dist("Deterministic")(
                self.recurrent_size, self.action_space.shape[0], noise=self.noise)
            self.scale = Scale(self.action_space)
            self.unscale = Unscale(self.action_space)
        else:
            raise NotImplementedError

        # ---- 5. Concatenate all policy modules ------------------------------

        policy_net = nn.Sequential(OrderedDict([
            ('obs_feature_extractor', policy_obs_feature_extractor),
            ('common_feature_extractor', policy_common_feature_extractor),
            ('memory_net', policy_memory_net), ('dist', dist),
        ]))

        setattr(self, name, policy_net)
