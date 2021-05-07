import gym
import torch
import torch.nn as nn
import numpy as np

import pytorchrl as prl
from pytorchrl.agent.actors.distributions import get_dist
from pytorchrl.agent.actors.utils import Scale, Unscale, init
from pytorchrl.agent.actors.memory_networks import GruNet
from pytorchrl.agent.actors.feature_extractors import MLP, default_feature_extractor


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
    noise : str
        Type of exploration noise that will be added to the deterministic actions.
    deterministic : bool
        Whether using DDPG, TD3 or any other deterministic off-policy actor.
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
    create_double_q_critic : bool
        Whether to instantiate a second Q network or not.

    Examples
    --------
    """
    def __init__(self,
                 input_space,
                 action_space,
                 noise=None,
                 deterministic=False,
                 sequence_overlap=0.5,
                 recurrent_nets_kwargs={},
                 recurrent_nets=False,
                 obs_feature_extractor=None,
                 obs_feature_extractor_kwargs={},
                 act_feature_extractor=None,
                 act_feature_extractor_kwargs={},
                 common_feature_extractor=MLP,
                 common_feature_extractor_kwargs={},
                 create_double_q_critic=True):

        """
        This actor defines policy network as:
        -------------------------------------

        policy = obs_feature_extractor + common_feature_extractor +

                + memory_net + action distribution

        and defines q networks as:
        -------------------------

            obs_feature_extractor
        q =                       + common_feature_extractor +
            act_feature_extractor

            + memory_net + q_prediction_layer

        """

        super(OffPolicyActor, self).__init__()

        self.input_space = input_space
        self.action_space = action_space
        self.deterministic = deterministic
        self.recurrent_nets = recurrent_nets
        self.sequence_overlap = np.clip(sequence_overlap, 0.0, 1.0)

        #######################################################################
        #                           POLICY NETWORK                            #
        #######################################################################

        # ---- 1. Define Obs feature extractor --------------------------------

        if len(input_space.shape) == 3:  # If inputs are images, CNN required
            obs_feature_extractor = default_feature_extractor(input_space)
        obs_extractor = obs_feature_extractor or nn.Identity
        self.policy_obs_feature_extractor = obs_extractor(
            input_space, **obs_feature_extractor_kwargs)

        # ---- 2. Define Common feature extractor -----------------------------

        feature_size = int(np.prod(self.policy_obs_feature_extractor(
            torch.randn(1, *input_space.shape)).shape))
        self.policy_common_feature_extractor = common_feature_extractor(
            feature_size, **common_feature_extractor_kwargs)

        # ---- 3. Define memory network  --------------------------------------

        feature_size = int(np.prod(self.policy_common_feature_extractor(
            torch.randn(1, feature_size)).shape))
        self.recurrent_size = feature_size
        if recurrent_nets:
            self.policy_memory_net = GruNet(feature_size, **recurrent_nets_kwargs)
            feature_size = self.policy_memory_net.num_outputs
        else:
            self.policy_memory_net = nn.Identity()

        # ---- 4. Define action distribution ----------------------------------

        if isinstance(action_space, gym.spaces.Discrete):
            self.dist = get_dist("Categorical")(feature_size, action_space.n)
            self.scale = None
            self.unscale = None

        elif isinstance(action_space, gym.spaces.Box) and not deterministic:
            self.dist = get_dist("SquashedGaussian")(feature_size, action_space.shape[0])
            self.scale = Scale(action_space)
            self.unscale = Unscale(action_space)

        elif isinstance(action_space, gym.spaces.Box) and deterministic:
            self.dist = get_dist("Deterministic")(feature_size,
                action_space.shape[0], noise=noise)
            self.scale = Scale(action_space)
            self.unscale = Unscale(action_space)
        else:
            raise NotImplementedError

        # ---- 5. Concatenate all policy modules ------------------------------

        self.policy_net = nn.Sequential(
            self.policy_obs_feature_extractor,
            self.policy_common_feature_extractor,
            self.policy_memory_net, self.dist)

        #######################################################################
        #                             Q-NETWORKS                              #
        #######################################################################

        # ---- 1. Define action feature extractor -----------------------------

        act_extractor = act_feature_extractor or nn.Identity
        self.q1_act_feature_extractor = act_extractor(
            action_space, **act_feature_extractor_kwargs)

        # ---- 2. Define obs feature extractor -----------------------------

        self.q1_obs_feature_extractor = obs_extractor(
            input_space, **obs_feature_extractor_kwargs)
        obs_feature_size = int(np.prod(self.q1_obs_feature_extractor(
            torch.randn(1, *input_space.shape)).shape))

        # ---- 3. Define shared feature extractor -----------------------------

        if isinstance(action_space, gym.spaces.Discrete):
            act_feature_size = 0
            q_outputs = action_space.n

        elif isinstance(action_space, gym.spaces.Box):
            act_feature_size = int(np.prod(self.q1_act_feature_extractor(
                torch.randn(1, *action_space.shape)).shape)) if act_feature_extractor \
                else np.prod(action_space.shape)
            q_outputs = 1

        else:
            raise NotImplementedError

        feature_size = obs_feature_size + act_feature_size
        self.q1_common_feature_extractor = common_feature_extractor(
            feature_size, **common_feature_extractor_kwargs)

        # ---- 4. Define memory network ---------------------------------------

        feature_size = int(np.prod(self.q1_common_feature_extractor(
            torch.randn(1, feature_size)).shape))
        self.q1_memory_net = GruNet(feature_size, **recurrent_nets_kwargs) if\
            recurrent_nets else nn.Identity()
        feature_size = self.q1_memory_net.num_outputs if recurrent_nets\
            else feature_size

        # ---- 5. Define prediction layer -------------------------------------

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.q1_predictor = init_(nn.Linear(feature_size, q_outputs))

        # ---- 6. Concatenate all q1 net modules ------------------------------

        self.q1 = nn.Sequential(
            self.q1_obs_feature_extractor,
            self.q1_act_feature_extractor,
            self.q1_common_feature_extractor,
            self.q1_memory_net, self.q1_predictor)

        # ---- 7. If create_double_q_critic, duplicate q1 net -----------------

        if create_double_q_critic:
            self.q2_obs_feature_extractor = obs_extractor(input_space, **obs_feature_extractor_kwargs)
            self.q2_act_feature_extractor = act_extractor(action_space, **act_feature_extractor_kwargs)
            feature_size = obs_feature_size + act_feature_size
            self.q2_common_feature_extractor = common_feature_extractor(feature_size, **common_feature_extractor_kwargs)
            feature_size = int(np.prod(self.q2_common_feature_extractor(torch.randn(1, feature_size)).shape))
            self.q2_memory_net = GruNet(feature_size, **recurrent_nets_kwargs) if recurrent_nets else nn.Identity()
            feature_size = self.q2_memory_net.num_outputs if recurrent_nets else feature_size
            self.q2_predictor = init_(nn.Linear(feature_size, q_outputs))
            self.q2 = nn.Sequential(
                self.q2_obs_feature_extractor,
                self.q2_act_feature_extractor,
                self.q2_common_feature_extractor,
                self.q1_memory_net, self.q2_predictor)

        else:
            self.q2_predictor = None
            self.q2 = torch.nn.Identity()


    @classmethod
    def create_factory(
            cls,
            input_space,
            action_space,
            noise=None,
            deterministic=False,
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
            create_double_q_critic=True

    ):
        """
        Returns a function that creates actor critic instances.

        Parameters
        ----------
        input_space : gym.Space
            Environment observation space.
        action_space : gym.Space
            Environment action space.
        noise : str
            Type of exploration noise that will be added to the deterministic actions.
        deterministic : bool
            Whether using DDPG, TD3 or any other deterministic off-policy actor.
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
        create_double_q_critic : bool
            Whether to instantiate a second Q network or not.

        Returns
        -------
        create_actor_critic_instance : func
            creates a new OffPolicyActor class instance.
        """

        def create_actor_critic_instance(device):
            """Create and return an actor critic instance."""
            policy = cls(input_space=input_space,
                         action_space=action_space,
                         noise=noise,
                         deterministic=deterministic,
                         sequence_overlap=sequence_overlap,
                         recurrent_nets_kwargs=recurrent_nets_kwargs,
                         recurrent_nets=recurrent_nets,
                         obs_feature_extractor=obs_feature_extractor,
                         obs_feature_extractor_kwargs=obs_feature_extractor_kwargs,
                         act_feature_extractor=act_feature_extractor,
                         act_feature_extractor_kwargs=act_feature_extractor_kwargs,
                         common_feature_extractor=common_feature_extractor,
                         common_feature_extractor_kwargs=common_feature_extractor_kwargs,
                         create_double_q_critic=create_double_q_critic)

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
        return self.policy_common_feature_extractor.recurrent_hidden_state_size

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

        rhs = {"rhs_act": rhs_act, "rhs_q1": rhs_act.clone(), "rhs_q2": rhs_act.clone()}

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

            if k.startswith(prl.RHS):
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

            act, _, _, rhs, _ = self.get_action(
                burn_in_data[prl.OBS], burn_in_data[prl.RHS], burn_in_data[prl.DONE])
            act2, _, _, rhs2, _ = self.get_action(
                burn_in_data[prl.OBS2], burn_in_data[prl.RHS2], burn_in_data[prl.DONE2])

            _, _, rhs = self.get_q_scores(
                burn_in_data[prl.OBS], rhs, burn_in_data[prl.DONE], act)
            _, _, rhs2 = self.get_q_scores(
                burn_in_data[prl.OBS2], rhs2, burn_in_data[prl.DONE2], act2)

            for k in rhs:
                rhs[k] = rhs[k].detach()
            for k in rhs2:
                rhs[k] = rhs[k].detach()

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
        """

        x = self.policy_common_feature_extractor(self.policy_obs_feature_extractor(obs))

        if self.recurrent_nets:
            x, rhs["rhs_act"] = self.policy_memory_net(x, rhs["rhs_act"], done)

        (action, clipped_action, logp_action, entropy_dist) = self.dist(
            x, deterministic=deterministic)

        if self.unscale:
            action = self.unscale(action)
            clipped_action = self.unscale(clipped_action)

        return action, clipped_action, logp_action, rhs, entropy_dist

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
        q1 : torch.tensor
            Q score according to current q1 network version.
        q2 : torch.tensor
            Q score according to current q2 network version.
        rhs : dict
            Updated recurrent hidden states.
        """

        features = self.q1_obs_feature_extractor(obs)
        if actions is not None:
            act_features = self.q1_act_feature_extractor(actions)
            features = torch.cat([features, act_features], -1)
        features = self.q1_common_feature_extractor(features)
        if self.recurrent_nets:
            features, rhs["rhs_q1"] = self.q1_memory_net(features, rhs["rhs_q1"], done)
        q1_scores = self.q1_predictor(features)

        if self.q2_predictor:
            features = self.q2_obs_feature_extractor(obs)
            if actions is not None:
                act_features = self.q2_act_feature_extractor(actions)
                features = torch.cat([features, act_features], -1)
            features = self.q2_common_feature_extractor(features)
            if self.recurrent_nets:
                features, rhs["rhs_q2"] = self.q2_memory_net(features, rhs["rhs_q2"], done)
            q2_scores = self.q2_predictor(features)

        else:
            q2_scores = None

        return q1_scores, q2_scores, rhs
