import torch
import torch.nn as nn
import torch.optim as optim

import pytorchrl as prl
from pytorchrl.utils import RunningMeanStd, clip_grad_norm_
from pytorchrl.agent.algorithms.base import Algorithm
from pytorchrl.agent.algorithms.policy_loss_addons import PolicyLossAddOn
from pytorchrl.agent.algorithms.utils import get_gradients, set_gradients
from pytorchrl.agent.actors.feature_extractors import default_feature_extractor


class RND_PPO(Algorithm):
    """
    Exploration by Random Network Distillation with Proximal Policy Optimization algorithm class.

    Algorithm class to execute RND PPO, from Burda et al., 2018
    (https://arxiv.org/abs/1810.12894). Algorithms are modules generally
    required by multiple workers, so RND_PPO.algo_factory(...) returns a function
    that can be passed on to workers to instantiate their own RND_PPO module.

    Parameters
    ----------
    device: torch.device
        CPU or specific GPU where class computations will take place.
    envs : VecEnv
        Vector of environments instance.
    actor : Actor
        Actor class instance.
    lr : float
        Optimizer learning rate.
    eps : float
        Optimizer epsilon parameter.
    num_epochs : int
        Number of PPO epochs.
    gamma : float
        Discount factor parameter.
    clip_param : float
        PPO clipping parameter.
    num_mini_batch : int
        Number of batches to create from collected data for actor updates.
    num_test_episodes : int
        Number of episodes to complete in each test phase.
    test_every : int
        Regularity of test evaluations.
    max_grad_norm : float
        Gradient clipping parameter.
    entropy_coef : float
        PPO entropy coefficient parameter.
    value_loss_coef : float
        PPO value coefficient parameter.
    use_clipped_value_loss : bool
        Prevent value loss from shifting too fast.
    policy_loss_addons : list
        List of PolicyLossAddOn components adding loss terms to the algorithm policy loss.
    gamma_intrinsic: float
        Discount factor parameter for intrinsic rewards.
    ext_adv_coeff: float
        Extrinsic advantage coefficient.
    int_adv_coeff: float
        Intrinsic advantage coefficient.
    predictor_proportion: float
        Proportion of buffer sample to use to train the predictor network.
    pre_normalization_steps: int
        Number of obs running average normalization steps to take before starting to train.
    pre_normalization_length: int
        Length of each pre normalization steps (in environment steps).
    intrinsic_rewards_network : nn.Module
        PyTorch nn.Module used for target and predictor networks.
    intrinsic_rewards_target_network_kwargs : dict
        Keyword arguments for the target network.
    intrinsic_rewards_predictor_network_kwargs : dict
        Keyword arguments for the predictor network.

    Examples
    --------
    >>> create_algo = RND_PPO.create_factory(
        lr=0.01, eps=1e-5, num_epochs=4, clip_param=0.2,
        entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5,
        num_mini_batch=4, use_clipped_value_loss=True, gamma=0.99)
    """

    def __init__(self,
                 envs,
                 actor,
                 device,
                 lr=1e-4,
                 eps=1e-8,
                 gamma=0.99,
                 num_epochs=4,
                 clip_param=0.2,
                 num_mini_batch=1,
                 test_every=1000,
                 max_grad_norm=2.0,
                 entropy_coef=0.01,
                 value_loss_coef=0.5,
                 num_test_episodes=5,
                 gamma_intrinsic=0.99,
                 ext_adv_coeff=2.0,
                 int_adv_coeff=1.0,
                 predictor_proportion=2.0,
                 pre_normalization_steps=50,
                 pre_normalization_length=128,
                 use_clipped_value_loss=False,
                 intrinsic_rewards_network=None,
                 intrinsic_rewards_target_network_kwargs={},
                 intrinsic_rewards_predictor_network_kwargs={},
                 policy_loss_addons=[]):

        # ---- General algo attributes ----------------------------------------

        # Discount factor
        self._gamma = gamma

        # Number of steps collected with initial random policy
        self._start_steps = 0  # Default to 0 for On-policy algos

        # Times data in the buffer is re-used before data collection proceeds
        self._num_epochs = int(num_epochs)

        # Number of data samples collected between network update stages
        self._update_every = None  # Depends on storage capacity

        # Number mini batches per epoch
        self._num_mini_batch = int(num_mini_batch)

        # Size of update mini batches
        self._mini_batch_size = None  # Depends on storage capacity

        # Number of network updates between test evaluations
        self._test_every = int(test_every)

        # Number of episodes to complete when testing
        self._num_test_episodes = int(num_test_episodes)

        # ---- RND-PPO-specific attributes ----------------------------------------

        self.envs = envs
        self.actor = actor
        self.device = device
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.value_loss_coef = value_loss_coef
        self.use_clipped_value_loss = use_clipped_value_loss

        self.ext_adv_coeff = ext_adv_coeff
        self.int_adv_coeff = int_adv_coeff
        self.gamma_intrinsic = gamma_intrinsic
        self.predictor_proportion = predictor_proportion
        self.pre_normalization_steps = pre_normalization_steps
        self.pre_normalization_length = pre_normalization_length

        assert hasattr(self.actor, "value_net1"), "RND_PPO requires value critic"
        assert hasattr(self.actor, "ivalue_net1"), "RND_PPO requires ivalue critic"

        # Get observation shape
        obs_space = self.envs.observation_space.shape

        # Get frame stack value
        frame_stack = 1
        if "frame_stack" in self.envs.env_kwargs.keys():
            frame_stack = self.envs.env_kwargs["frame_stack"]

        # Get number of obs channels
        self.obs_channels = int(obs_space[0] / frame_stack)

        # Define network type
        int_net = intrinsic_rewards_network or default_feature_extractor(self.envs.observation_space)

        # Create target model
        setattr(
            self.actor, "target_model",
            int_net((self.obs_channels,) + obs_space[1:],
                    **intrinsic_rewards_target_network_kwargs).to(self.device))

        # Freeze target model parameters
        for param in self.actor.target_model.parameters():
            param.requires_grad = False

        # Create predictor model
        setattr(
            self.actor, "predictor_model",
            int_net((self.obs_channels,) + obs_space[1:],
                    **intrinsic_rewards_predictor_network_kwargs).to(self.device))

        # Define running means for int reward and obs
        self.state_rms = RunningMeanStd(shape=(1, ) + obs_space[1:], device=self.device)

        print("---Pre_normalization started.---")
        obs, rhs, done = self.actor.actor_initial_states(envs.reset())
        total_obs = torch.zeros(
            (self.pre_normalization_length,  obs.shape[0], self.obs_channels) + obs.shape[2:]).to(self.device)
        for i in range(self.pre_normalization_steps * self.pre_normalization_length):
            _, clipped_action, rhs, _ = self.acting_step(obs, rhs, done)
            obs, _, _, _ = envs.step(clipped_action)
            total_obs[i % self.pre_normalization_length].copy_(obs[:, -self.obs_channels:, ...])
            if i % self.pre_normalization_length == 0 and i != 0:
                self.state_rms.update(total_obs.reshape(-1, *total_obs.shape[2:]))
                print("{}/{}".format(i//self.pre_normalization_length, self.pre_normalization_steps))
        envs.reset()
        print("---Pre_normalization is done.---")

        # ----- Optimizers ----------------------------------------------------

        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr, eps=eps)

        # ----- Policy Loss Addons --------------------------------------------

        # Sanity check, policy_loss_addons is a PolicyLossAddOn instance
        # or a list of PolicyLossAddOn instances
        assert isinstance(policy_loss_addons, (PolicyLossAddOn, list)),\
            "RND PPO policy_loss_addons parameter should be a  PolicyLossAddOn instance " \
            "or a list of PolicyLossAddOn instances"
        if isinstance(policy_loss_addons, list):
            for addon in policy_loss_addons:
                assert isinstance(addon, PolicyLossAddOn), \
                    "RND PPO policy_loss_addons parameter should be a  PolicyLossAddOn " \
                    "instance or a list of PolicyLossAddOn instances"
        else:
            policy_loss_addons = [policy_loss_addons]

        self.policy_loss_addons = policy_loss_addons
        for addon in self.policy_loss_addons:
            addon.setup(self.device)

    @classmethod
    def create_factory(cls,
                       lr=1e-4,
                       eps=1e-8,
                       gamma=0.99,
                       num_epochs=4,
                       clip_param=0.2,
                       num_mini_batch=1,
                       test_every=1000,
                       max_grad_norm=0.5,
                       entropy_coef=0.01,
                       value_loss_coef=0.5,
                       num_test_episodes=5,
                       gamma_intrinsic=0.99,
                       ext_adv_coeff=2.0,
                       int_adv_coeff=1.0,
                       predictor_proportion=2.0,
                       pre_normalization_steps=50,
                       pre_normalization_length=128,
                       use_clipped_value_loss=True,
                       intrinsic_rewards_network=None,
                       intrinsic_rewards_target_network_kwargs={},
                       intrinsic_rewards_predictor_network_kwargs={},
                       policy_loss_addons=[]):
        """
        Returns a function to create new RND PPO instances.

        Parameters
        ----------
        lr : float
            Optimizer learning rate.
        eps : float
            Optimizer epsilon parameter.
        num_epochs : int
            Number of PPO epochs.
        gamma : float
            Discount factor parameter.
        clip_param : float
            PPO clipping parameter.
        num_mini_batch : int
            Number of batches to create from collected data for actor update.
        num_test_episodes : int
            Number of episodes to complete in each test phase.
        test_every : int
            Regularity of test evaluations.
        max_grad_norm : float
            Gradient clipping parameter.
        entropy_coef : float
            PPO entropy coefficient parameter.
        value_loss_coef : float
            PPO value coefficient parameter.
        use_clipped_value_loss : bool
            Prevent value loss from shifting too fast.
        gamma_intrinsic: float
            Discount factor parameter for intrinsic rewards.
        ext_adv_coeff: float
            Extrinsic advantage coefficient.
        int_adv_coeff: float
            Intrinsic advantage coefficient.
        predictor_proportion: float
            Proportion of buffer sample to use to train the predictor network.
        pre_normalization_steps: int
            Number of obs running average normalization steps to take before starting to train.
        pre_normalization_length: int
            Length of each pre normalization steps (in environment steps).
        intrinsic_rewards_network : nn.Module
            PyTorch nn.Module used for target and predictor networks.
        intrinsic_rewards_target_network_kwargs : dict
            Keyword arguments for the target network.
        intrinsic_rewards_predictor_network_kwargs : dict
            Keyword arguments for the predictor network.
        policy_loss_addons : list
            List of PolicyLossAddOn components adding loss terms to the algorithm policy loss.

        Returns
        -------
        create_algo_instance : func
            Function that creates a new PPO class instance.
        algo_name : str
            Name of the algorithm.
        """
        def create_algo_instance(device, actor, envs):
            return cls(lr=lr,
                       eps=eps,
                       envs=envs,
                       actor=actor,
                       gamma=gamma,
                       device=device,
                       test_every=test_every,
                       num_epochs=num_epochs,
                       clip_param=clip_param,
                       entropy_coef=entropy_coef,
                       ext_adv_coeff=ext_adv_coeff,
                       int_adv_coeff=int_adv_coeff,
                       max_grad_norm=max_grad_norm,
                       num_mini_batch=num_mini_batch,
                       value_loss_coef=value_loss_coef,
                       gamma_intrinsic=gamma_intrinsic,
                       num_test_episodes=num_test_episodes,
                       predictor_proportion=predictor_proportion,
                       pre_normalization_length=pre_normalization_length,
                       pre_normalization_steps=pre_normalization_steps,
                       use_clipped_value_loss=use_clipped_value_loss,
                       intrinsic_rewards_network=intrinsic_rewards_network,
                       intrinsic_rewards_target_network_kwargs=intrinsic_rewards_target_network_kwargs,
                       intrinsic_rewards_predictor_network_kwargs=intrinsic_rewards_predictor_network_kwargs,
                       policy_loss_addons=policy_loss_addons)

        return create_algo_instance, prl.RND_PPO

    @property
    def gamma(self):
        """Returns discount factor gamma."""
        return self._gamma

    @property
    def start_steps(self):
        """Returns the number of steps to collect with initial random policy."""
        return self._start_steps

    @property
    def num_epochs(self):
        """
        Returns the number of times the whole buffer is re-used before data
        collection proceeds.
        """
        return self._num_epochs

    @property
    def update_every(self):
        """
        Returns the number of data samples collected between
        network update stages.
        """
        return self._update_every

    @property
    def num_mini_batch(self):
        """
        Returns the number of times the whole buffer is re-used before data
        collection proceeds.
        """
        return self._num_mini_batch

    @property
    def mini_batch_size(self):
        """
        Returns the number of mini batches per epoch.
        """
        return self._mini_batch_size

    @property
    def test_every(self):
        """Number of network updates between test evaluations."""
        return self._test_every

    @property
    def num_test_episodes(self):
        """
        Returns the number of episodes to complete when testing.
        """
        return self._num_test_episodes

    def acting_step(self, obs, rhs, done, deterministic=False):
        """
        PPO acting function.

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
            Additional PPO predictions, value score and action log probability.
        """

        with torch.no_grad():

            (action, clipped_action, logp_action, rhs,
             entropy_dist, dist) = self.actor.get_action(
                obs, rhs, done, deterministic)

            value_dict = self.actor.get_value(obs, rhs, done)
            ext_value = value_dict.pop("value_net1")
            int_value = value_dict.pop("ivalue_net1")
            rhs = value_dict.pop("rhs")

            # predict intrinsic reward
            obs = obs[:, -self.obs_channels:, ...]
            obs = torch.clamp((obs - self.state_rms.mean.float()) / (self.state_rms.var.float() ** 0.5), -5, 5)
            # obs = torch.clamp((obs - self.state_rms.mean) / (self.state_rms.var ** 0.5), -5, 5).float()
            predictor_encoded_features = self.actor.predictor_model(obs)
            target_encoded_features = self.actor.target_model(obs)
            int_reward = (predictor_encoded_features - target_encoded_features).pow(2).mean(1).unsqueeze(1)

            other = {prl.VAL: ext_value, prl.IVAL: int_value, prl.LOGP: logp_action, prl.IREW: int_reward}

        return action, clipped_action, rhs, other

    def compute_loss(self, data):
        """
        Compute PPO loss from data batch.

        Parameters
        ----------
        data: dict
            Data batch dict containing all required tensors to compute PPO loss.

        Returns
        -------
        value_loss: torch.tensor
            value term of PPO loss.
        action_loss: torch.tensor
            policy term of PPO loss.
        dist_entropy: torch.tensor
            policy term of PPO loss.
        loss: torch.tensor
            PPO loss.
        """

        o, rhs, a, old_v = data[prl.OBS], data[prl.RHS], data[prl.ACT], data[prl.VAL]
        r, d, old_logp, adv = data[prl.RET], data[prl.DONE], data[prl.LOGP], data[prl.ADV]

        # RDN PPO
        ir, old_iv, iadv = data[prl.IRET], data[prl.IVAL], data[prl.IADV]
        mask = data[prl.MASK] if prl.MASK in data.keys() else None

        advs = adv * self.ext_adv_coeff + iadv * self.int_adv_coeff

        new_logp, dist_entropy, dist = self.actor.evaluate_actions(o, rhs, d, a)

        new_vs = self.actor.get_value(o, rhs, d)
        new_v = new_vs.get("value_net1")
        new_iv = new_vs.get("ivalue_net1")

        # Policy loss
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advs
        action_loss = - torch.min(surr1, surr2).mean()

        # Ext value loss
        if self.use_clipped_value_loss:
            # Ext value
            value_losses = (new_v - r).pow(2)
            value_pred_clipped = old_v + (new_v - old_v).clamp(-self.clip_param, self.clip_param)
            value_losses_clipped = (value_pred_clipped - r).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            # Int value
            ivalue_losses = (new_iv - ir).pow(2)
            ivalue_pred_clipped = old_iv + (new_iv - old_iv).clamp(-self.clip_param, self.clip_param)
            ivalue_losses_clipped = (ivalue_pred_clipped - ir).pow(2)
            ivalue_loss = 0.5 * torch.max(ivalue_losses, ivalue_losses_clipped).mean()
        else:
            # Ext value
            value_loss = 0.5 * (r - new_v).pow(2).mean()
            # Int value
            ivalue_loss = 0.5 * (ir - new_iv).pow(2).mean()

        total_value_loss = value_loss + ivalue_loss

        o = o[:, -self.obs_channels:, ...]
        o = torch.clamp((o - self.state_rms.mean.float()) / (self.state_rms.var.float() ** 0.5), -5, 5)

        # Rnd loss
        encoded_target_features = self.actor.target_model(o)
        encoded_predictor_features = self.actor.predictor_model(o)
        loss = (encoded_predictor_features - encoded_target_features).pow(2).mean(-1)
        mask2 = torch.rand(loss.size(), device=self.device)
        mask2 = (mask2 <= self.predictor_proportion).float()
        mask = mask.squeeze() * mask2 if mask is not None else mask2
        rnd_loss = (mask * loss).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))

        loss = total_value_loss * self.value_loss_coef + action_loss - self.entropy_coef * dist_entropy + rnd_loss

        # Extend policy loss with addons
        for addon in self.policy_loss_addons:
            loss += addon.compute_loss_term(self.actor, dist, data)

        return value_loss, ivalue_loss, action_loss, rnd_loss, dist_entropy, loss

    def compute_gradients(self, batch, grads_to_cpu=True):
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
            List of actor gradients.
        info: dict
            Dict containing current PPO iteration information.
        """

        value_loss, ivalue_loss, action_loss, rnd_loss, dist_entropy, loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()

        clip_grad_norm_(self.actor.parameters())
        # nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

        pi_grads = get_gradients(self.actor.policy_net, grads_to_cpu=grads_to_cpu)
        v_grads = get_gradients(self.actor.value_net1, grads_to_cpu=grads_to_cpu)
        iv_grads = get_gradients(self.actor.ivalue_net1, grads_to_cpu=grads_to_cpu)
        pred_grads = get_gradients(self.actor.predictor_model, grads_to_cpu=grads_to_cpu)

        grads = {"pi_grads": pi_grads, "v_grads": v_grads, "iv_grads": iv_grads, "pred_grads": pred_grads}

        info = {
            "loss": loss.item(),
            "value_loss": value_loss.item(),
            "ivalue_loss": ivalue_loss.item(),
            "rnd_loss": rnd_loss.item(),
            "action_loss": action_loss.item(),
            "entropy_loss": dist_entropy.item(),
            "mean_intrinsic_rewards": batch[prl.IREW].mean().cpu().item(),
            "min_intrinsic_rewards": batch[prl.IREW].min().cpu().item(),
            "max_intrinsic_rewards": batch[prl.IREW].max().cpu().item(),
        }

        return grads, info

    def apply_gradients(self, gradients=None):
        """
        Take an optimization step, previously setting new gradients if provided.

        Parameters
        ----------
        gradients: list of tensors
            List of actor gradients.
        """
        if gradients is not None:
            set_gradients(
                self.actor.policy_net,
                gradients=gradients["pi_grads"], device=self.device)
            set_gradients(
                self.actor.value_net1,
                gradients=gradients["v_grads"], device=self.device)
            set_gradients(
                self.actor.ivalue_net1,
                gradients=gradients["iv_grads"], device=self.device)
            set_gradients(
                self.actor.predictor_model,
                gradients=gradients["pred_grads"], device=self.device)

        self.optimizer.step()

    def set_weights(self, actor_weights):
        """
        Update actor with the given weights

        Parameters
        ----------
        actor_weights: dict of tensors
            Dict containing actor weights to be set.
        """
        self.actor.load_state_dict(actor_weights)

    def update_algorithm_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of the algorithm, change its value
        to `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Worker.algo attribute name
        new_parameter_value : int or float
            New value for `parameter_name`.
        """
        if hasattr(self, parameter_name):
            setattr(self, parameter_name, new_parameter_value)
        if parameter_name == "lr":
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_parameter_value
