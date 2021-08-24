import gym
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

import pytorchrl as prl
from pytorchrl.agent.actors.distributions import get_dist
from pytorchrl.agent.actors.utils import Scale, Unscale, init, partially_load_checkpoint
from pytorchrl.agent.actors.feature_extractors import MLP, default_feature_extractor
from pytorchrl.agent.actors.planner import MPC

class MBActor(nn.Module):
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
    algorithm : str
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

    Examples
    --------
    """
    def __init__(self,
                 input_space,
                 action_space,
                 algorithm,
                 env_name,
                 n_planner,
                 depth,
                 noise=None,
                 obs_feature_extractor=None,
                 obs_feature_extractor_kwargs={},
                 act_feature_extractor=None,
                 act_feature_extractor_kwargs={},
                 common_feature_extractor=MLP,
                 common_feature_extractor_kwargs={},):

        super(MBActor, self).__init__()

        self.noise = noise
        self.algorithm = algorithm
        self.input_space = input_space
        self.action_space = action_space
        self.act_feature_extractor = act_feature_extractor
        self.act_feature_extractor_kwargs = act_feature_extractor_kwargs
        self.obs_feature_extractor = obs_feature_extractor
        self.obs_feature_extractor_kwargs = obs_feature_extractor_kwargs
        self.common_feature_extractor = common_feature_extractor
        self.common_feature_extractor_kwargs = common_feature_extractor_kwargs

        self.create_dynamics()
        
        self.mpc = MPC(env_name=env_name,
                       action_space=self.action_space,
                       n_planner=n_planner,
                       depth=depth)

    @classmethod
    def create_factory(
            cls,
            input_space,
            action_space,
            env_name,
            n_planner,
            depth,
            algorithm,
            noise=None,
            restart_model=None,
            obs_feature_extractor=None,
            obs_feature_extractor_kwargs={},
            act_feature_extractor=None,
            act_feature_extractor_kwargs={},
            common_feature_extractor=MLP,
            common_feature_extractor_kwargs={},


    ):
        """
        Returns a function that creates actor critic instances.

        Parameters
        ----------
        input_space : gym.Space
            Environment observation space.
        action_space : gym.Space
            Environment action space.
        algorithm : str
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

        Returns
        -------
        create_actor_critic_instance : func
            creates a new OffPolicyActor class instance.
        """

        def create_actor_critic_instance(device):
            """Create and return an actor critic instance."""
            model = cls(input_space=input_space,
                         action_space=action_space,
                         env_name=env_name,
                         n_planner=n_planner,
                         depth=depth,
                         algorithm=algorithm,
                         noise=noise,
                         obs_feature_extractor=obs_feature_extractor,
                         obs_feature_extractor_kwargs=obs_feature_extractor_kwargs,
                         act_feature_extractor=act_feature_extractor,
                         act_feature_extractor_kwargs=act_feature_extractor_kwargs,
                         common_feature_extractor=common_feature_extractor,
                         common_feature_extractor_kwargs=common_feature_extractor_kwargs)

            if isinstance(restart_model, str):
                model.load_state_dict(torch.load(restart_model, map_location=device))
            elif isinstance(restart_model, dict):
                for submodule, checkpoint in restart_model.items():
                    partially_load_checkpoint(model, submodule, checkpoint)
            model.to(device)

            return model

        return create_actor_critic_instance

    def get_action(self, obs, deterministic=False):
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

        return self.mpc.get_next_action(obs, self.dynamics_model, deterministic)

    def create_dynamics(self, name="dynamics_model"):
        """
        This actor defines defines dynamics networks as:

            obs_feature_extractor
        q =                       + common_feature_extractor + q_prediction_layer
            act_feature_extractor

        """

        # ---- 1. Define action feature extractor -----------------------------

        act_extractor = self.act_feature_extractor or nn.Identity
        dynamics_act_feature_extractor = act_extractor(
            self.action_space, **self.act_feature_extractor_kwargs)

        # ---- 2. Define obs feature extractor -----------------------------

        obs_extractor = self.obs_feature_extractor or nn.Identity
        dynamics_obs_feature_extractor = obs_extractor(
            self.input_space, **self.obs_feature_extractor_kwargs)
        obs_feature_size = int(np.prod(dynamics_obs_feature_extractor(
            torch.randn(1, *self.input_space.shape)).shape))

        # ---- 3. Define shared feature extractor -----------------------------

        if isinstance(self.action_space, gym.spaces.Discrete):
            act_feature_size = 0

        elif isinstance(self.action_space, gym.spaces.Box):
            act_feature_size = int(np.prod(dynamics_act_feature_extractor(
                torch.randn(1, *self.action_space.shape)).shape)) if self.act_feature_extractor \
                else np.prod(self.action_space.shape)
        else:
            raise NotImplementedError

        feature_size = obs_feature_size + act_feature_size
        dynamics_common_feature_extractor = self.common_feature_extractor(
            feature_size, **self.common_feature_extractor_kwargs)

        # ---- 3. Output Layer
        logits = get_dist("Deterministic")(
            feature_size, self.input_space.shape[0], noise=self.noise)
        self.scale = Scale(self.action_space)
        self.unscale = Unscale(self.action_space)


        # ---- 6. Concatenate all q1 net modules ------------------------------

        dynamics_net = nn.Sequential(OrderedDict([
            ('obs_feature_extractor', dynamics_obs_feature_extractor),
            ('act_feature_extractor', dynamics_act_feature_extractor),
            ('common_feature_extractor', dynamics_common_feature_extractor),
            ("output_logits", logits),
        ]))

        setattr(self, name, dynamics_net)
