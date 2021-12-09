import itertools
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.kl import kl_divergence

import pytorchrl as prl
from pytorchrl.agent.algorithms.base import Algorithm
from pytorchrl.agent.algorithms.utils import get_gradients, set_gradients, gaussian_kl
from pytorchrl.agent.algorithms.policy_loss_addons import PolicyLossAddOn


class MPO(Algorithm):
    """
    Maximum a Posteriori Policy Optimization algorithm class.

    Algorithm class to execute MPO, from  A Abdolmaleki et al.
    (https://arxiv.org/abs/1806.06920). Algorithms are modules generally
    required by multiple workers, so MPO.algo_factory(...) returns a function
    that can be passed on to workers to instantiate their own MPO module.

    This code has been adapted from https://github.com/daisatojp/mpo.

    Parameters
    ----------
    device : torch.device
        CPU or specific GPU where class computations will take place.
    envs : VecEnv
        Vector of environments instance.
    actor : Actor
        Actor_critic class instance.
    lr_pi : float
        Policy optimizer learning rate.
    lr_q : float
        Q-nets optimizer learning rate.
    gamma : float
        Discount factor parameter.
    polyak : float
        SAC polyak averaging parameter.
    num_updates : int
        Num consecutive actor updates before data collection continues.
    update_every : int
        Regularity of actor updates in number environment steps.
    start_steps : int
        Num of initial random environment steps before learning starts.
    mini_batch_size : int
        Size of actor update batches.
    target_update_interval : float
        regularity of target nets updates with respect to actor Adam updates.
    num_test_episodes : int
        Number of episodes to complete in each test phase.
    test_every : int
        Regularity of test evaluations in actor updates.
    dual_constraint : float
        Hard constraint of the dual formulation in the E-step corresponding to [2] p.4 ε.
    kl_mean_constraint : float
        Hard constraint of the mean in the M-step corresponding to [2] p.6 ε_μ for continuous action space.
    kl_var_constraint : float
        Hard constraint of the covariance in the M-step corresponding to [2] p.6 ε_Σ for continuous action space.
    kl_constraint : float
        Hard constraint in the M-step corresponding to [2] p.6 ε_π for discrete action space.
    alpha_scale: float
        Scaling factor of the lagrangian multiplier in the M-step for dicrete action spaces.
    alpha_max : float
        Higher bound used for clipping the lagrangian lagrangian in discrete action spaces.
    alpha_mean_scale : float
        Mean scaling factor of the lagrangian multiplier in the M-step for continuous action spaces.
    alpha_var_scale : float
        Varience scaling factor of the lagrangian lagrangian in the M-step for continuous action spaces.
    alpha_mean_max : float
        Higher bound used for clipping the lagrangian lagrangian in continuous action spaces.
    alpha_var_max : float
        Higher bound used for clipping the lagrangian lagrangian in continuous action spaces.
    mstep_iterations : int
        The number of iterations of the M-step
    sample_action_num : int
        For continuous action spaces, number of samples used to compute expected Q scores.
    max_grad_norm : float
        Gradient clipping parameter.
    policy_loss_addons : list
        List of PolicyLossAddOn components adding loss terms to the algorithm policy loss.

    Examples
    --------
    >>> create_algo = MPO.create_factory(
            lr_q=1e-4, lr_pi=1e-4, lr_alpha=1e-4, gamma=0.99, polyak=0.995,
            num_updates=50, update_every=50, test_every=5000, start_steps=20000,
            mini_batch_size=64, alpha=1.0, num_test_episodes=0, target_update_interval=1)
    """

    def __init__(self,
                 device,
                 envs,
                 actor,
                 lr_q=1e-4,
                 lr_pi=1e-4,
                 gamma=0.99,
                 polyak=1.0,
                 num_updates=1,
                 update_every=50,
                 test_every=1000,
                 start_steps=20000,
                 mini_batch_size=64,
                 num_test_episodes=5,
                 target_update_interval=1,
                 dual_constraint=0.1,
                 kl_mean_constraint=0.01,
                 kl_var_constraint=0.0001,
                 kl_constraint=0.01,
                 alpha_scale=10.0,
                 alpha_mean_scale=1.0,
                 alpha_var_scale=100.0,
                 alpha_mean_max=0.1,
                 alpha_var_max=10.0,
                 alpha_max=1.0,
                 mstep_iterations=5,
                 sample_action_num=64,
                 max_grad_norm=0.1,
                 policy_loss_addons=[]):

        # ---- General algo attributes ----------------------------------------

        # Discount factor
        self._gamma = gamma

        # Number of steps collected with initial random policy
        self._start_steps = int(start_steps)

        # Times data in the buffer is re-used before data collection proceeds
        self._num_epochs = int(1)  # Default to 1 for off-policy algorithms

        # Number of data samples collected between network update stages
        self._update_every = int(update_every)

        # Number mini batches per epoch
        self._num_mini_batch = int(num_updates)

        # Size of update mini batches
        self._mini_batch_size = int(mini_batch_size)

        # Number of network updates between test evaluations
        self._test_every = int(test_every)

        # Number of episodes to complete when testing
        self._num_test_episodes = int(num_test_episodes)

        # ---- MPO-specific attributes ----------------------------------------

        self.iter = 0
        self.envs = envs
        self.actor = actor
        self.device = device
        self.polyak = polyak
        self.dual_constraint = dual_constraint
        self.max_grad_norm = max_grad_norm
        self.mstep_iterations = mstep_iterations
        self.sample_action_num = sample_action_num
        self.target_update_interval = target_update_interval

        # For continuous action space
        self.kl_mean_constraint = kl_mean_constraint
        self.kl_var_constraint = kl_var_constraint
        self.alpha_mean_scale = alpha_mean_scale
        self.alpha_var_scale = alpha_var_scale
        self.alpha_mean_max = alpha_mean_max
        self.alpha_var_max = alpha_var_max

        # For discrete action space
        self.alpha_max = alpha_max
        self.kl_constraint = kl_constraint
        self.alpha_scale = alpha_scale

        # Initialize Lagrange Multiplier
        self.eta = np.random.rand()
        self.alpha_mu = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.alpha_sigma = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.alpha = 0.0  # lagrangian multiplier for discrete action space in the M-step

        assert hasattr(self.actor, "q1"), "MPO requires q critic (num_critics=1)"

        # Create target networks
        self.actor_targ = deepcopy(actor)

        # Freeze target networks with respect to optimizers
        for p in self.actor_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks
        q_params = self.actor.q1.parameters()

        # List of parameters for policy network
        p_params = itertools.chain(self.actor.policy_net.parameters())

        # ----- Optimizers ----------------------------------------------------

        self.norm_loss_q = nn.SmoothL1Loss()
        self.pi_optimizer = optim.Adam(p_params, lr=lr_pi)
        self.q_optimizer = optim.Adam(q_params, lr=lr_q)

        # ----- Policy Loss Addons --------------------------------------------

        # Sanity check, policy_loss_addons is a PolicyLossAddOn instance
        # or a list of PolicyLossAddOn instances
        assert isinstance(policy_loss_addons, (PolicyLossAddOn, list)),\
            "MPO policy_loss_addons parameter should be a  PolicyLossAddOn instance " \
            "or a list of PolicyLossAddOn instances"
        if isinstance(policy_loss_addons, list):
            for addon in policy_loss_addons:
                assert isinstance(addon, PolicyLossAddOn), \
                    "MPO policy_loss_addons parameter should be a  PolicyLossAddOn " \
                    "instance or a list of PolicyLossAddOn instances"
        else:
            policy_loss_addons = [policy_loss_addons]

        self.policy_loss_addons = policy_loss_addons
        for addon in self.policy_loss_addons:
            addon.setup(self.device)

    @classmethod
    def create_factory(cls,
                       lr_q=1e-4,
                       lr_pi=1e-4,
                       gamma=0.99,
                       polyak=0.995,
                       num_updates=50,
                       test_every=5000,
                       update_every=50,
                       start_steps=1000,
                       mini_batch_size=64,
                       num_test_episodes=5,
                       target_update_interval=1.0,
                       dual_constraint=0.1,
                       kl_mean_constraint=0.01,
                       kl_var_constraint=0.0001,
                       kl_constraint=0.01,
                       alpha_scale=10.0,
                       alpha_mean_scale=1.0,
                       alpha_var_scale=100.0,
                       alpha_mean_max=0.1,
                       alpha_var_max=10.0,
                       alpha_max=1.0,
                       mstep_iterations=5,
                       sample_action_num=64,
                       max_grad_norm=0.1,
                       policy_loss_addons=[]):
        """
        Returns a function to create new MPO instances.

        Parameters
        ----------
        lr_pi : float
            Policy optimizer learning rate.
        lr_q : float
            Q-nets optimizer learning rate.
        gamma : float
            Discount factor parameter.
        polyak : float
            SAC polyak averaging parameter.
        num_updates : int
            Num consecutive actor updates before data collection continues.
        update_every : int
            Regularity of actor updates in number environment steps.
        start_steps : int
            Num of initial random environment steps before learning starts.
        mini_batch_size : int
            Size of actor update batches.
        target_update_interval : float
            regularity of target nets updates with respect to actor Adam updates.
        num_test_episodes : int
            Number of episodes to complete in each test phase.
        test_every : int
            Regularity of test evaluations in actor updates.
        dual_constraint : float
            Hard constraint of the dual formulation in the E-step corresponding to [2] p.4 ε.
        kl_mean_constraint : float
            Hard constraint of the mean in the M-step corresponding to [2] p.6 ε_μ for continuous action space.
        kl_var_constraint : float
            Hard constraint of the covariance in the M-step corresponding to [2] p.6 ε_Σ for continuous action space.
        kl_constraint : float
            Hard constraint in the M-step corresponding to [2] p.6 ε_π for discrete action space.
        alpha_scale: float
            Scaling factor of the lagrangian multiplier in the M-step for dicrete action spaces.
        alpha_max : float
            Higher bound used for clipping the lagrangian lagrangian in discrete action spaces.
        alpha_mean_scale : float
            Mean scaling factor of the lagrangian multiplier in the M-step for continuous action spaces.
        alpha_var_scale : float
            Varience scaling factor of the lagrangian lagrangian in the M-step for continuous action spaces.
        alpha_mean_max : float
            Higher bound used for clipping the lagrangian lagrangian in continuous action spaces.
        alpha_var_max : float
            Higher bound used for clipping the lagrangian lagrangian in continuous action spaces.
        mstep_iterations : int
            The number of iterations of the M-step
        sample_action_num : int
            For continuous action spaces, number of samples used to compute expected Q scores.
        max_grad_norm : float
            Gradient clipping parameter.
        policy_loss_addons : list
            List of PolicyLossAddOn components adding loss terms to the algorithm policy loss.

        Returns
        -------
        create_algo_instance : func
            Function that creates a new MPO class instance.
        algo_name : str
            Name of the algorithm.
        """

        def create_algo_instance(device, actor, envs):
            return cls(lr_q=lr_q,
                       lr_pi=lr_pi,
                       envs=envs,
                       actor=actor,
                       gamma=gamma,
                       device=device,
                       polyak=polyak,
                       test_every=test_every,
                       start_steps=start_steps,
                       num_updates=num_updates,
                       update_every=update_every,
                       mini_batch_size=mini_batch_size,
                       num_test_episodes=num_test_episodes,
                       target_update_interval=target_update_interval,
                       dual_constraint=dual_constraint,
                       kl_mean_constraint=kl_mean_constraint,
                       kl_var_constraint=kl_var_constraint,
                       kl_constraint=kl_constraint,
                       alpha_scale=alpha_scale,
                       alpha_mean_scale=alpha_mean_scale,
                       alpha_var_scale=alpha_var_scale,
                       alpha_mean_max=alpha_mean_max,
                       alpha_var_max=alpha_var_max,
                       policy_loss_addons=policy_loss_addons)

        return create_algo_instance, prl.MPO

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

    @property
    def discrete_version(self):
        """Returns True if action_space is discrete."""
        return self.actor.action_space.__class__.__name__ == "Discrete"

    def acting_step(self, obs, rhs, done, deterministic=False):
        """
        MPO acting function.

        Parameters
        ----------
        obs : torch.tensor
            Current world observation
        rhs : torch.tensor
            RNN recurrent hidden state (if policy is not a RNN, rhs will contain zeroes).
        done : torch.tensor
            1.0 if current obs is the last one in the episode, else 0.0.
        deterministic : bool
            Whether to randomly sample action from predicted distribution or taking the mode.

        Returns
        -------
        action : torch.tensor
            Predicted next action.
        clipped_action : torch.tensor
            Predicted next action (clipped to be within action space).
        rhs : torch.tensor
            Policy recurrent hidden state (if policy is not a RNN, rhs will contain zeroes).
        other : dict
            Additional MPO predictions, which are not used in other algorithms.
        """

        with torch.no_grad():
            (action, clipped_action, logp_action, rhs,
             entropy_dist, dist) = self.actor.get_action(
                obs, rhs, done, deterministic=deterministic)

        return action, clipped_action, rhs, {}

    def compute_loss_q(self, batch, n_step=1, per_weights=1):
        """
        Calculate MPO Q-nets loss

        Parameters
        ----------
        batch : dict
            Data batch dict containing all required tensors to compute MPO losses.
        n_step : int or float
            Number of future steps used to computed the truncated n-step return value.
        per_weights :
            Prioritized Experience Replay (PER) important sampling weights or 1.0.

        Returns
        -------
        loss_q : torch.tensor
            Q-net loss.
        errors : torch.tensor
            TD errors.
        """

        # [2] 3 Policy Evaluation (Step 1)

        o, rhs, d = batch[prl.OBS], batch[prl.RHS], batch[prl.DONE]
        a, r = batch[prl.ACT], batch[prl.REW]
        o2, rhs2, d2 = batch[prl.OBS2], batch[prl.RHS2], batch[prl.DONE2]

        bs, ds = o.shape[0], o.shape[-1]

        if self.discrete_version:

            # Q-values for all actions
            q1 = self.actor.get_q_scores(o, rhs, d).get("q1")
            q1 = q1.gather(1, a.long())

            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                a2, _, _, _, _, dist = self.actor_targ.get_action(o2, rhs2, d2)

                N = dist.probs.shape[-1]  # num actions
                actions = torch.arange(N)[..., None].expand(-1, bs).to(self.device)  # (da, bs)
                p_a2 = dist.expand((N, bs)).log_prob(actions).exp().transpose(0, 1)  # (bs, da)

                # Target Q-values
                q1_pi_targ = self.actor_targ.get_q_scores(o2, rhs2, d2).get("q1")
                q_pi_targ = (p_a2 * q1_pi_targ).sum(dim=1, keepdim=True)

                assert r.shape == q_pi_targ.shape
                backup = r + (self.gamma ** n_step) * (1 - d2) * q_pi_targ

        else:

            N = self.sample_action_num
            da = a.shape[-1]  # num action dimensions

            # Q-values for all actions
            q1 = self.actor.get_q_scores(o, rhs, d, a).get("q1")

            # Bellman backup for Q functions
            with torch.no_grad():

                # Target actions come from *current* policy
                a2, _, logp_a2, _, _, dist = self.actor_targ.get_action(o2, rhs2, d2)

                sampled_actions = dist.sample((N,)).transpose(0, 1)  # (bs, N, da)
                expanded_obs2 = o2[:, None, :].expand(-1, N, -1)  # (bs, N, ds)
                expanded_d2 = d2[:, None, :].expand(-1, N, -1)  # (bs, N, 1)
                expanded_rhs2 = {k: v[:, None, :].expand(-1, N, -1) for k, v in rhs2.items()}
                expanded_reshaped_rhs2 = {k: v.reshape(-1, v.shape[-1]) for k, v in expanded_rhs2.items()}

                next_q1 = self.actor_targ.get_q_scores(
                    expanded_obs2.reshape(-1, ds),  # (N * bs, ds)
                    expanded_reshaped_rhs2,  # get expanded rhs
                    expanded_d2.reshape(-1, 1),  # (N * bs, ds)
                    sampled_actions.reshape(-1, da),  # (N * bs, da)
                ).get("q1")  # (N * bs, 1)

                expected_next_q1 = next_q1.reshape(bs, N).mean(dim=1, keepdim=True)  # (B,)
                q_pi_targ = expected_next_q1

                backup = r + (self.gamma ** n_step) * (1 - d2) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = 0.5 * (((q1 - backup) ** 2) * per_weights).mean()
        # loss_q = self.norm_loss_q(backup, q1)

        errors = ((q1 - backup).abs()).detach().cpu()

        return loss_q, errors

    def compute_loss_pi(self, batch, per_weights=1):
        """
        Calculate MPO policy loss.

        Parameters
        ----------
        batch : dict
            Data batch dict containing all required tensors to compute MPO losses.
        per_weights :
            Prioritized Experience Replay (PER) important sampling weights or 1.0.

        Returns
        -------
        loss_policy : torch.tensor
            MPO policy loss.
        """

        o, rhs, d, a = batch[prl.OBS], batch[prl.RHS], batch[prl.DONE], batch[prl.ACT]
        bs, ds = o.shape[0], o.shape[-1]

        # E-Step of Policy Improvement
        # [2] 4.1 Finding action weights (Step 2)

        _, _, _, _, _, dist_targ = self.actor_targ.get_action(o, rhs, d)

        if self.discrete_version:

            N = dist_targ.probs.shape[-1]  # num possible actions actions, env.action_space.n

            # for each state in the batch, any possible action
            actions = torch.arange(N)[..., None].expand(N, bs).to(self.device)  # (N, bs)
            dist_targ_probs = dist_targ.expand((N, bs)).log_prob(actions).exp()  # (N, bs)

            target_q1 = self.actor_targ.get_q_scores(o, rhs, d).get("q1")  # (bs, N)
            target_q1 = target_q1.transpose(1, 0)  # (N, bs)

            b_prob_np = dist_targ_probs.cpu().transpose(0, 1).numpy()  # (bs, N)
            target_q1_np = target_q1.cpu().transpose(0, 1).numpy()  # (bs, N)

        else:

            N = self.sample_action_num
            da = a.shape[-1]  # num action dimensions
            sampled_actions = dist_targ.sample((N,))  # (N, bs, da)
            expanded_obs = o[None, ...].expand(N, -1, -1)  # (N, bs, ds)
            expanded_d = d[None, ...].expand(N, -1, -1)  # (N, bs, 1)
            expanded_rhs = {k: v[None, ...].expand(N, -1, -1) for k, v in rhs.items()}
            expanded_reshaped_rhs = {k: v.reshape(-1, v.shape[-1]) for k, v in expanded_rhs.items()}

            target_q1 = self.actor_targ.get_q_scores(
                expanded_obs.reshape(-1, ds),  # (N * bs, ds)
                expanded_reshaped_rhs,  # get expanded rhs
                expanded_d.reshape(-1, 1),  # (N * bs, ds)
                sampled_actions.reshape(-1, da),  # (N * bs, da)
            ).get("q1")

            target_q1 = target_q1.reshape(N, bs)  # (N, bs)
            target_q1_np = target_q1.cpu().transpose(0, 1).numpy()  # (bs, N)

        # https://arxiv.org/pdf/1812.02256.pdf
        # [2] 4.1 Finding action weights (Step 2)
        #   Using an exponential transformation of the Q-values
        if self.discrete_version:
            def dual(eta):
                """
                dual function of the non-parametric variational
                g(η) = η*ε + η*mean(log(sum(π(a|s)*exp(Q(s, a)/η))))
                We have to multiply π by exp because this is expectation.
                This equation is correspond to last equation of the [2] p.15
                For numerical stabilization, this can be modified to
                Qj = max(Q(s, a), along=a)
                g(η) = η*ε + mean(Qj, along=j) + η*mean(log(sum(π(a|s)*(exp(Q(s, a)-Qj)/η))))
                """
                max_q = np.max(target_q1_np, 1)
                return eta * self.dual_constraint + np.mean(max_q) + eta * np.mean(np.log(np.sum(
                    b_prob_np * np.exp((target_q1_np - max_q[:, None]) / eta), axis=1)))
        else:  # discrete action space
            def dual(eta):
                """
                dual function of the non-parametric variational
                Q = target_q_np  (K, N)
                g(η) = η*ε + η*mean(log(mean(exp(Q(s, a)/η), along=a)), along=s)
                For numerical stabilization, this can be modified to
                Qj = max(Q(s, a), along=a)
                g(η) = η*ε + mean(Qj, along=j) + η*mean(log(mean(exp((Q(s, a)-Qj)/η), along=a)), along=s)
                """
                max_q = np.max(target_q1_np, 1)
                return eta * self.dual_constraint + np.mean(max_q) + eta * np.mean(np.log(
                    np.mean(np.exp((target_q1_np - max_q[:, None]) / eta), axis=1)))

        bounds = [(1e-6, None)]
        res = minimize(dual, np.array([self.eta]), method='SLSQP', bounds=bounds)
        self.eta = res.x[0]
        qij = torch.softmax(target_q1 / self.eta, dim=0)  # (N, bs)

        # M-Step of Policy Improvement
        # [2] 4.2 Fitting an improved policy (Step 3)
        for _ in range(self.mstep_iterations):
            if self.discrete_version:

                _, _, _, _, _, dist = self.actor.get_action(o, rhs, d)
                loss_pi = torch.mean(qij * dist.expand((N, bs)).log_prob(actions))
                kl = kl_divergence(dist, dist_targ).mean()

                # Update lagrange multipliers by gradient descent
                # this equation is derived from last eq of [2] p.5,
                # just differentiate with respect to α
                # and update α so that the equation is to be minimized.
                self.alpha -= self.alpha_scale * (self.kl_constraint - kl).detach().item()
                self.alpha = np.clip(self.alpha, 0.0, self.alpha_max)

                # last eq of [2] p.5
                loss_policy = -(loss_pi + self.alpha * (self.kl_constraint - kl))

            else:

                _, _, _, _, _, dist = self.actor.get_action(o, rhs, d)

                loss_pi = torch.mean(
                    qij * (
                            dist_targ.expand((N, bs, da)).log_prob(sampled_actions).sum(-1)  # (N, K)
                            + dist.expand((N, bs, da)).log_prob(sampled_actions).sum(-1)  # (N, K)
                    )
                )

                # Define diag covariance matrices
                cov1 = torch.eye(dist.variance.shape[-1]).to(self.device) * dist.variance.unsqueeze(
                    2).expand(*dist.variance.size(), dist.variance.size(1))
                cov2 = torch.eye(dist_targ.variance.shape[-1]).to(self.device) * dist_targ.variance.unsqueeze(
                    2).expand(*dist_targ.variance.size(), dist_targ.variance.size(1))
                kl_mu, kl_sigma = gaussian_kl(dist_targ.mean, dist.mean, cov1, cov2)

                if np.isnan(kl_mu.item()):  # This should not happen
                    raise RuntimeError('kl_mu is nan')
                if np.isnan(kl_sigma.item()):  # This should not happen
                    raise RuntimeError('kl_sigma is nan')

                # Update lagrange multipliers by gradient descent
                # this equation is derived from last eq of [2] p.5, just differentiate with
                # respect to α and update α so that the equation is to be minimized.
                self.alpha_mu -= self.alpha_mean_scale * (self.kl_mean_constraint - kl_mu).detach().item()
                self.alpha_sigma -= self.alpha_var_scale * (self.kl_var_constraint - kl_mu).detach().item()

                self.alpha_mu = np.clip(self.alpha_mu, 0.0, self.alpha_mean_max)
                self.alpha_sigma = np.clip(self.alpha_sigma, 0.0, self.alpha_var_max)

                # last eq of [2] p.5
                loss_policy = -(loss_pi + self.alpha_mu * (self.kl_mean_constraint - kl_mu) +
                                self.alpha_sigma * (self.kl_var_constraint - kl_sigma))

        # Extend policy loss with addons
        for addon in self.policy_loss_addons:
           loss_policy += addon.compute_loss_term(self.actor, dist, batch)

        return loss_policy

    def compute_gradients(self, batch, grads_to_cpu=True):
        """
        Compute loss and compute gradients but don't do optimization step,
        return gradients instead.

        Parameters
        ----------
        batch : dict
            data batch containing all required tensors to compute MPO losses.
        grads_to_cpu : bool
            If gradient tensor will be sent to another node, need to be in CPU.

        Returns
        -------
        grads : list of tensors
            List of actor gradients.
        info : dict
            Dict containing current MPO iteration information.
        """

        # Recurrent burn-in
        if self.actor.is_recurrent:
            batch = self.actor.burn_in_recurrent_states(batch)

        # N-step returns
        n_step = batch["n_step"] if "n_step" in batch else 1.0

        # PER
        per_weights = batch["per_weights"] if "per_weights" in batch else 1.0

        # Run one gradient descent step for Q1 and Q2
        loss_q, errors = self.compute_loss_q(batch, n_step, per_weights)
        self.q_optimizer.zero_grad()
        loss_q.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.q1.parameters(), self.max_grad_norm)
        q_grads = get_gradients(self.actor.q1, self.actor.q2, grads_to_cpu=grads_to_cpu)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.actor.q1.parameters():
            p.requires_grad = False

        # Run one gradient descent step for pi.
        loss_pi = self.compute_loss_pi(batch, per_weights)

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        nn.utils.clip_grad_norm_(self.actor.policy_net.parameters(), self.max_grad_norm)
        pi_grads = get_gradients(self.actor.policy_net, grads_to_cpu=grads_to_cpu)

        for p in self.actor.q1.parameters():
            p.requires_grad = True

        info = {
            "loss_q": loss_q.detach().item(),
            "loss_pi": loss_pi.detach().item(),
        }

        if "per_weights" in batch:
            info.update({"errors": errors})

        grads = {"q_grads": q_grads, "pi_grads": pi_grads}

        return grads, info

    def update_target_networks(self):
        """Update actor critic target networks with polyak averaging."""
        if self.iter % self.target_update_interval == 0:
            with torch.no_grad():
                for p, p_targ in zip(self.actor.parameters(), self.actor_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def apply_gradients(self, gradients=None):
        """
        Take an optimization step, previously setting new gradients if provided.
        Update also target networks.

        Parameters
        ----------
        gradients : list of tensors
            List of actor gradients.
        """
        if gradients:
            set_gradients(
                self.actor.policy_net,
                gradients=gradients["pi_grads"], device=self.device)
            set_gradients(
                self.actor.q1, self.actor.q2,
                gradients=gradients["q_grads"], device=self.device)

        self.q_optimizer.step()
        self.pi_optimizer.step()

        # Update target networks by polyak averaging.
        self.iter += 1
        self.update_target_networks()

    def set_weights(self, actor_weights):
        """
        Update actor with the given weights. Update also target networks.

        Parameters
        ----------
        actor_weights : dict of tensors
            Dict containing actor weights to be set.
        """
        self.actor.load_state_dict(actor_weights)

        # Update target networks by polyak averaging.
        self.iter += 1
        self.update_target_networks()

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
            for param_group in self.pi_optimizer.param_groups:
                param_group['lr'] = new_parameter_value
            for param_group in self.q_optimizer.param_groups:
                param_group['lr'] = new_parameter_value
