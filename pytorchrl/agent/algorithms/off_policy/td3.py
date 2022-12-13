import itertools
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

import pytorchrl as prl
from pytorchrl.agent.algorithms.base import Algorithm
from pytorchrl.agent.algorithms.utils import get_gradients, set_gradients
from pytorchrl.agent.algorithms.policy_loss_addons import PolicyLossAddOn


class TD3(Algorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradient algorithm class.
    Algorithm class to execute TD3, from Scott Fujimoto et al.
    Addressing Function Approximation Error in Actor-Critic Methods
    (https://arxiv.org/pdf/1802.09477.pdf).

    Algorithms are modules generally required by multiple workers, so TD3.algo_factory(...)
    returns a function that can be passed on to workers to instantiate their own TD3 module.

    Parameters
    ----------
    device : torch.device
        CPU or specific GPU where class computations will take place.
    envs : VecEnv
        Vector of environments instance.
    actor : Actor
        Actor class instance.
    lr_pi : float
        Policy optimizer learning rate.
    lr_q : float
        Q-nets optimizer learning rate.
    gamma : float
        Discount factor parameter.
    polyak : float
        TD3 polyak averaging parameter.
    num_updates : int
        Num consecutive actor_critic updates before data collection continues.
    update_every : int
        Regularity of actor_critic updates in number environment steps.
    start_steps : int
        Num of initial random environment steps before learning starts.
    mini_batch_size : int
        Size of actor_critic update batches.
    target_update_interval: float
        regularity of target nets updates with respect to actor_critic Adam updates.
    num_test_episodes : int
        Number of episodes to complete in each test phase.
    test_every : int
        Regularity of test evaluations in actor_critic updates.
    max_grad_norm : float
        Gradient clipping parameter.
    policy_loss_addons : list
        List of PolicyLossAddOn components adding loss terms to the algorithm policy loss.

    Examples
    --------
    >>> create_algo = TD3.create_factory(
            lr_q=1e-3, lr_pi=1e-3, gamma=0.99, polyak=0.995,
            num_updates=50, update_every=50, test_every=5000, start_steps=20000,
            mini_batch_size=100, num_test_episodes=0, target_update_interval=2)
    """

    def __init__(self,
                 device,
                 envs,
                 actor,
                 lr_q=1e-4,
                 lr_pi=1e-4,
                 gamma=0.99,
                 polyak=0.995,
                 num_updates=1,
                 update_every=50,
                 test_every=1000,
                 max_grad_norm=0.5,
                 start_steps=20000,
                 mini_batch_size=64,
                 num_test_episodes=5,
                 target_update_interval=1,
                 policy_loss_addons=[]):

        # ---- General algo attributes ----------------------------------------

        # Discount factor
        self._gamma = gamma

        # Number of steps collected with initial random policy
        self._start_steps = int(start_steps)

        # Times data in the buffer is re-used before data collection proceeds
        self._num_epochs = 1  # Default to 1 for off-policy algorithms

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

        # ---- TD3-specific attributes ----------------------------------------

        self.iter = 0
        self.envs = envs
        self.actor = actor
        self.prev_loss_pi = torch.FloatTensor([0.])
        self.polyak = polyak
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.target_update_interval = target_update_interval

        self.action_low = self.actor.action_space.low[0]  # Can sometimes be a vector?
        self.action_high = self.actor.action_space.high[0]

        assert hasattr(self.actor, "q1"), "TD3 requires double q critic (num_critics=2)"
        assert hasattr(self.actor, "q2"), "TD3 requires double q critic (num_critics=2)"

        # Create target networks
        self.actor_targ = deepcopy(actor)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks
        q_params = itertools.chain(self.actor.q1.parameters(), self.actor.q2.parameters())

        # List of parameters for both Q-networks
        p_params = itertools.chain(self.actor.policy_net.parameters())

        # ----- Policy Loss Addons --------------------------------------------

        # Sanity check, policy_loss_addons is a PolicyLossAddOn instance
        # or a list of PolicyLossAddOn instances
        assert isinstance(policy_loss_addons, (PolicyLossAddOn, list)),\
            "TD3 policy_loss_addons parameter should be a  PolicyLossAddOn instance " \
            "or a list of PolicyLossAddOn instances"
        if isinstance(policy_loss_addons, list):
            for addon in policy_loss_addons:
                assert isinstance(addon, PolicyLossAddOn), \
                    "TD3 policy_loss_addons parameter should be a  PolicyLossAddOn" \
                    " instance or a list of PolicyLossAddOn instances"
        else:
            policy_loss_addons = [policy_loss_addons]

        self.policy_loss_addons = policy_loss_addons
        for addon in self.policy_loss_addons:
            addon.setup(self.actor, self.device)

        # ----- Optimizers ----------------------------------------------------

        self.pi_optimizer = optim.Adam(p_params, lr=lr_pi)
        self.q_optimizer = optim.Adam(q_params, lr=lr_q)

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
                       max_grad_norm=0.5,
                       mini_batch_size=100,
                       num_test_episodes=5,
                       target_update_interval=1.0,
                       policy_loss_addons=[]):
        """
        Returns a function to create new TD3 instances.

        Parameters
        ----------
        lr_pi : float
            Policy optimizer learning rate.
        lr_q : float
            Q-nets optimizer learning rate.
        gamma : float
            Discount factor parameter.
        polyak : float
            TD3 polyak averaging parameter.
        num_updates : int
            Num consecutive actor_critic updates before data collection continues.
        update_every : int
            Regularity of actor_critic updates in number environment steps.
        start_steps : int
            Num of initial random environment steps before learning starts.
        mini_batch_size : int
            Size of actor_critic update batches.
        target_update_interval : float
            regularity of target nets updates with respect to actor_critic Adam updates.
        num_test_episodes : int
            Number of episodes to complete in each test phase.
        test_every : int
            Regularity of test evaluations in actor_critic updates.
        max_grad_norm : float
            Gradient clipping parameter.
        policy_loss_addons : list
            List of PolicyLossAddOn components adding loss terms to the algorithm policy loss.

        Returns
        -------
        create_algo_instance : func
            Function that creates a new TD3 class instance.
        algo_name : str
            Name of the algorithm.
        """

        def create_algo_instance(device, actor, envs):
            return cls(lr_q=lr_q,
                       envs=envs,
                       actor=actor,
                       lr_pi=lr_pi,
                       gamma=gamma,
                       device=device,
                       polyak=polyak,
                       test_every=test_every,
                       start_steps=start_steps,
                       num_updates=num_updates,
                       update_every=update_every,
                       max_grad_norm=max_grad_norm,
                       mini_batch_size=mini_batch_size,
                       num_test_episodes=num_test_episodes,
                       target_update_interval=target_update_interval,
                       policy_loss_addons=policy_loss_addons)

        return create_algo_instance, prl.TD3

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
        TD3 acting function.

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
        clipped_action: torch.tensor
            Predicted next action (clipped to be within action space).
        rhs : torch.tensor
            Policy recurrent hidden state (if policy is not a RNN, rhs will contain zeroes).
        other : dict
            Additional TD3 predictions, which are not used in other algorithms.
        """

        with torch.no_grad():
            (action, clipped_action, logp_action, rhs,
             entropy_dist, dist) = self.actor.get_action(
                obs, rhs, done, deterministic=deterministic)

        return action, clipped_action, rhs, {}

    def compute_loss_q(self, data, n_step=1, per_weights=1):
        """
        Calculate TD3 Q-nets loss

        Parameters
        ----------
        data: dict
            Data batch dict containing all required tensors to compute TD3 losses.
        n_step : int or float
            Number of future steps used to computed the truncated n-step return value.
        per_weights :
            Prioritized Experience Replay (PER) important sampling weights or 1.0.

        Returns
        -------
        loss_q1 : torch.tensor
            Q1-net loss.
        loss_q2 : torch.tensor
            Q2-net loss.
        loss_q : torch.tensor
            Weighted average of loss_q1 and loss_q2.
        errors : torch.tensor
            TD errors.
        """
        o, rhs, d = data[prl.OBS], data[prl.RHS], data[prl.DONE]
        a, r = data[prl.ACT], data[prl.REW]
        o2, rhs2, d2 = data[prl.OBS2], data[prl.RHS2], data[prl.DONE2]

        # Q-values for all actions
        q_scores = self.actor.get_q_scores(o, rhs, d, a)
        q1 = q_scores.get("q1")
        q2 = q_scores.get("q2")

        # Bellman backup for Q functions
        with torch.no_grad():

            # Target actions come from *current* policy
            a2, _, _, _, _, dist = self.actor.get_action(o2, rhs2, d2)
            noise = torch.clamp(torch.normal(mean=torch.FloatTensor(
                [0.0]), std=torch.FloatTensor([0.2]) ).to(a2.device), min=-0.5, max=0.5)
            a2 = torch.clamp(a2 + noise, min=self.action_low, max=self.action_high)

            # Target Q-values
            q_scores_targ = self.actor_targ.get_q_scores(o2, rhs2, d2, a2)
            q1_targ = q_scores_targ.get("q1")
            q2_targ = q_scores_targ.get("q2")
            q_pi_targ = torch.min(q1_targ, q2_targ)

            backup = r + (self.gamma ** n_step) * (1 - d2) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = (((q1 - backup) ** 2) * per_weights).mean()
        loss_q2 = (((q2 - backup) ** 2) * per_weights).mean()
        loss_q = 0.5 * loss_q1 + 0.5 * loss_q2

        errors = (torch.min(q1, q2) - backup).abs().detach().cpu()
        # errors = torch.max((q1 - backup).abs(), (q2 - backup).abs()).detach().cpu()

        # reset Noise
        self.actor.policy_net.dist.noise.reset()

        return loss_q1, loss_q2, loss_q, errors

    def compute_loss_pi(self, data, per_weights=1):
        """
        Calculate TD3 policy loss.

        Parameters
        ----------
        data: dict
            Data batch dict containing all required tensors to compute TD3 losses.

        Returns
        -------
        loss_pi : torch.tensor
            TD3 policy loss.
        """

        o, rhs, d = data[prl.OBS], data[prl.RHS], data[prl.DONE]

        pi, _, _, _, _, dist = self.actor.get_action(o, rhs, d)
        q_scores = self.actor.get_q_scores(o, rhs, d, pi)
        q1_pi = q_scores.get("q1")

        # q_pi = torch.min(q1_pi, q2_pi) # commenting this out since the paper only
        # uses q1 but might be worth testing if using min gives general improvement

        loss_pi = - (q1_pi * per_weights).mean()

        # Extend policy loss with addons
        addons_info = {}
        for addon in self.policy_loss_addons:
            addon_loss, addons_info = addon.compute_loss_term(batch, dist, addons_info)
            loss_pi += addon_loss

        return loss_pi, addons_info

    def compute_gradients(self, batch, grads_to_cpu=True):
        """
        Compute loss and compute gradients but don't do optimization step,
        return gradients instead.

        Parameters
        ----------
        batch: dict
            data batch containing all required tensors to compute TD3 losses.
        grads_to_cpu: bool
            If gradient tensor will be sent to another node, need to be in CPU.

        Returns
        -------
        grads: list of tensors
            List of actor_critic gradients.
        info: dict
            Dict containing current TD3 iteration information.
        """

        # Recurrent burn-in
        if self.actor.is_recurrent:
            batch = self.actor.burn_in_recurrent_states(batch)

        # PER
        per_weights = batch.pop("per_weights") if "per_weights" in batch else 1.0

        # N-step returns
        n_step = batch.pop("n_step") if "n_step" in batch else 1.0

        # First run one gradient descent step for Q1 and Q2
        loss_q1, loss_q2, loss_q, errors = self.compute_loss_q(batch, n_step, per_weights)
        self.q_optimizer.zero_grad()
        loss_q.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(itertools.chain(
            self.actor.q1.parameters(), self.actor.q2.parameters()), self.max_grad_norm)
        q_grads = get_gradients(self.actor.q1, self.actor.q2, grads_to_cpu=grads_to_cpu)
        grads = {"q_grads": q_grads}

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in itertools.chain(self.actor.q1.parameters(), self.actor.q2.parameters()):
            p.requires_grad = False

        # Compute policy
        loss_pi = self.prev_loss_pi
        if self.iter % self.target_update_interval == 0:
            loss_pi, addons_info = self.compute_loss_pi(batch, per_weights)
            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.prev_loss_pi = loss_pi
            nn.utils.clip_grad_norm_(self.actor.policy_net.parameters(), self.max_grad_norm)
            pi_grads = get_gradients(self.actor.policy_net, grads_to_cpu=grads_to_cpu)
            grads.update({"pi_grads": pi_grads})

        for p in itertools.chain(self.actor.q1.parameters(), self.actor.q2.parameters()):
            p.requires_grad = True

        info = {
            "loss_q1": loss_q1.detach().item(),
            "loss_q2": loss_q2.detach().item(),
            "loss_pi": loss_pi.detach().item(),
        }

        if "per_weights" in batch:
            info.update({"errors": errors})

        info.update(addons_info)

        return grads, info

    def update_target_networks(self):
        """Update actor critic target networks with polyak averaging"""
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
        if gradients is not None:
            set_gradients(
                self.actor.q1, self.actor.q2,
                gradients=gradients["q_grads"], device=self.device)
            if "pi_grads" in gradients.keys():
                set_gradients(
                    self.actor.policy_net,
                    gradients=gradients["pi_grads"], device=self.device)

        self.q_optimizer.step()
        if self.iter % self.target_update_interval == 0:
            self.pi_optimizer.step()
            self.update_target_networks()

        # Update target networks by polyak averaging.
        self.iter += 1


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
