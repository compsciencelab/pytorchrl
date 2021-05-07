import torch
import torch.nn as nn
import torch.optim as optim

import pytorchrl as prl
from pytorchrl.agent.algorithms.base import Algorithm


class PPO(Algorithm):
    """
    Proximal Policy Optimization algorithm class.

    Algorithm class to execute PPO, from Schulman et al.
    (https://arxiv.org/abs/1707.06347). Algorithms are modules generally
    required by multiple workers, so PPO.algo_factory(...) returns a function
    that can be passed on to workers to instantiate their own PPO module.

    Parameters
    ----------
    device: torch.device
        CPU or specific GPU where class computations will take place.
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

    Examples
    --------
    >>> create_algo = PPO.create_factory(
        lr=0.01, eps=1e-5, num_epochs=4, clip_param=0.2,
        entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5,
        num_mini_batch=4, use_clipped_value_loss=True, gamma=0.99)
    """

    def __init__(self,
                 device,
                 actor,
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
                 use_clipped_value_loss=True):

        # ---- General algo attributes ----------------------------------------

        # Discount factor
        self._gamma = gamma

        # Number of steps collected with initial random policy
        self._start_steps = 0  # Default to 0 for On-policy algos

        # Times data in the buffer is re-used before data collection proceeds
        self._num_epochs = num_epochs

        # Number of data samples collected between network update stages
        self._update_every = None  # Depends on storage capacity

        # Number mini batches per epoch
        self._num_mini_batch = num_mini_batch

        # Size of update mini batches
        self._mini_batch_size = None  # Depends on storage capacity

        # Number of network updates between test evaluations
        self._test_every = test_every

        # Number of episodes to complete when testing
        self._num_test_episodes = num_test_episodes

        # ---- PPO-specific attributes ----------------------------------------

        self.device = device
        self.actor = actor
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.value_loss_coef = value_loss_coef
        self.use_clipped_value_loss = use_clipped_value_loss

        # ----- Optimizers ----------------------------------------------------

        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr, eps=eps)

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
                     use_clipped_value_loss=True):
        """
        Returns a function to create new PPO instances.

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

        Returns
        -------
        create_algo_instance : func
            creates a new PPO class instance.
        """
        def create_algo_instance(device, actor):
            return cls(lr=lr,
                       eps=eps,
                       gamma=gamma,
                       device=device,
                       actor=actor,
                       test_every=test_every,
                       num_epochs=num_epochs,
                       clip_param=clip_param,
                       entropy_coef=entropy_coef,
                       max_grad_norm=max_grad_norm,
                       num_mini_batch=num_mini_batch,
                       value_loss_coef=value_loss_coef,
                       num_test_episodes=num_test_episodes,
                       use_clipped_value_loss=use_clipped_value_loss)

        return create_algo_instance

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
             entropy_dist) = self.actor.get_action(
                obs, rhs, done, deterministic)

            value, rhs = self.actor.get_value(obs, rhs, done)

            other = {prl.VAL: value, prl.LOGP: logp_action}

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

        new_logp, dist_entropy, _ = self.actor.evaluate_actions(o, rhs, d, a)
        new_v, _ = self.actor.get_value(o, rhs, d)

        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
        action_loss = - torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_losses = (new_v - r).pow(2)
            value_pred_clipped = old_v + (new_v - old_v).clamp(-self.clip_param, self.clip_param)
            value_losses_clipped = (value_pred_clipped - r).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (r - new_v).pow(2).mean()

        loss = value_loss * self.value_loss_coef + action_loss - self.entropy_coef * dist_entropy

        return value_loss, action_loss, dist_entropy, loss

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

        self.optimizer.zero_grad()
        value_loss, action_loss, dist_entropy, loss  = self.compute_loss(batch)

        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

        grads = []
        for p in self.actor.parameters():
            if grads_to_cpu:
                if p.grad is not None: grads.append(p.grad.data.cpu().numpy())
                else: grads.append(None)
            else:
                if p.grad is not None:
                    grads.append(p.grad)

        info = {
            "loss": loss.item(),
            "value_loss": value_loss.item(),
            "action_loss": action_loss.item(),
            "entropy_loss": dist_entropy.item()
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
        if gradients:
            for g, p in zip(gradients, self.actor.parameters()):
                if g is not None:
                    p.grad = torch.from_numpy(g).to(self.device)
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

