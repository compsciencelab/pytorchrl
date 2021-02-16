import torch
import torch.nn as nn
import torch.optim as optim

from .base import Algo


class A2C(Algo):
    """
    Algorithm class to execute A2C, from Mnih et al. 2016 (https://arxiv.org/pdf/1602.01783.pdf).

    Parameters
    ----------
    device : torch.device
        CPU or specific GPU where class computations will take place.
    actor_critic : ActorCritic
        Actor_critic class instance.
    lr_v : float
        Value network learning rate.
    lr_pi : float
        Policy network learning rate.
    gamma : float
        Discount factor parameter.
    num_test_episodes : int
        Number of episodes to complete in each test phase.
    max_grad_norm : float
        Gradient clipping parameter.
    test_every : int
        Regularity of test evaluations in actor_critic updates.
    num_test_episodes : int
        Number of episodes to complete in each test phase.

    Attributes
    ----------
    device : torch.device
        CPU or specific GPU where class computations will take place.
    actor_critic : ActorCritic
        ActorCritic Class containing Neural Network function approximators.
    test_every : int
        Number of network updates between test evaluations.
    num_test_episodes : int
        Num episodes to complete in each test phase.
    gamma : float
        Discount factor parameter.
    max_grad_norm : float
        Gradient clipping parameter.
    pi_optimizer : torch.optimizer
        Policy network optimizer
    v_optimizer : torch.optimizer
        Value network optimizer
    """

    def __init__(self,
                 device,
                 actor_critic,
                 lr_v=1e-4,
                 lr_pi=1e-4,
                 gamma=0.99,
                 test_every=5000,
                 max_grad_norm=0.5,
                 num_test_episodes=5):

        # ---- General algo attributes ----------------------------------------

        # Number of steps collected with initial random policy
        self.start_steps = 0  # Default to 0 for On-policy algos

        # Times data in the buffer is re-used before data collection proceeds
        self.num_epochs = 1

        # Number of data samples collected between network update stages
        self.update_every = None  # Depends on storage capacity

        # Number mini batches per epoch
        self.num_mini_batch = 1

        # Size of update mini batches
        self.mini_batch_size = None  # Depends on storage capacity

        # Number of network updates between test evaluations
        self.test_every = test_every

        # Number of episodes to complete when testing
        self.num_test_episodes = num_test_episodes

        # ---- A2C-specific attributes ----------------------------------------

        self.iter = 0
        self.gamma = gamma
        self.device = device
        self.actor_critic = actor_critic
        self.max_grad_norm = max_grad_norm

        # ----- Optimizer -----------------------------------------------------

        self.pi_optimizer = optim.Adam(self.actor_critic.policy_net.parameters(), lr=lr_pi)
        self.v_optimizer = optim.Adam(self.actor_critic.value_net.parameters(), lr=lr_v)

    @classmethod
    def create_factory(cls,
                       lr_v=1e-4,
                       lr_pi=1e-4,
                       gamma=0.99,
                       test_every=5000,
                       max_grad_norm=0.5,
                       num_test_episodes=5):
        """
        Returns a function to create new A2C instances.

        Parameters
        ----------
        lr_v : float
            Value network learning rate.
        lr_pi : float
            Policy network learning rate.
        gamma : float
            Discount factor parameter.
        num_test_episodes : int
            Number of episodes to complete in each test phase.
        max_grad_norm : float
            Gradient clipping parameter.
        test_every : int
            Regularity of test evaluations in actor_critic updates.
        num_test_episodes : int
            Number of episodes to complete in each test phase.

        Returns
        -------
        create_algo_instance : func
            creates a new A2C class instance.
        """

        def create_algo_instance(device, actor):
            return cls(lr_pi=lr_pi,
                       lr_v=lr_v,
                       gamma=gamma,
                       device=device,
                       actor_critic=actor,
                       test_every=test_every,
                       max_grad_norm=max_grad_norm,
                       num_test_episodes=num_test_episodes)
        return create_algo_instance

    def acting_step(self, obs, rhs, done, deterministic=False):
        """
        A2C acting function.

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
            Additional A2C predictions, value score and action log probability.
        """

        with torch.no_grad():
            (action, clipped_action, logp_action, rhs,
             entropy_dist) = self.actor_critic.get_action(
                obs, rhs, done, deterministic)
            value = self.actor_critic.get_value(obs, rhs, done)
            other = {"val": value, "logp": logp_action}

        return action, clipped_action, rhs, other

    def compute_loss(self, data):
        """
        Calculate A2C loss

        Parameters
        ----------
        data: dict
            Data batch dict containing all required tensors to compute A2C loss.
        rnn_hs : torch.tensor
            Policy recurrent hidden state obtained with the current ActorCritic version.

        Returns
        -------
        loss : torch.tensor
            A2C loss.
        """

        o, rhs, a, old_v = data["obs"], data["rhs"], data["act"], data["val"]
        r, d, old_logp, adv = data["ret"], data["done"], data["logp"], data["adv"]

        # Policy loss
        logp, dist_entropy, _ = self.actor_critic.evaluate_actions(o, rhs, d, a)
        pi_loss = - (logp * adv).mean()

        # Value loss
        new_v = self.actor_critic.get_value(o, rhs, d)
        value_loss = (r - new_v).pow(2).mean()

        return pi_loss, value_loss

    def compute_gradients(self, batch, grads_to_cpu=True):
        """
        Compute loss and compute gradients but don't do optimization step,
        return gradients instead.

        Parameters
        ----------
        data: dict
            data batch containing all required tensors to compute A2C loss.
        grads_to_cpu: bool
            If gradient tensor will be sent to another node, need to be in CPU.

        Returns
        -------
        grads: list of tensors
            List of actor_critic gradients.
        info: dict
            Dict containing current A2C iteration information.
        """

        # Compute A2C losses
        action_loss, value_loss = self.compute_loss(batch)

        # Compute policy gradients
        self.pi_optimizer.zero_grad()
        action_loss.backward(retain_graph=True)

        for p in self.actor_critic.policy_net.parameters():
            p.requires_grad = False

        # Compute value gradients
        self.v_optimizer.zero_grad()
        value_loss.backward()

        for p in self.actor_critic.policy_net.parameters():
            p.requires_grad = True

        # Clip gradients to max value
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        grads = []
        for p in self.actor_critic.parameters():
            if grads_to_cpu:
                if p.grad is not None:
                    grads.append(p.grad.data.cpu().numpy())
                else:
                    grads.append(None)
            else:
                if p.grad is not None:
                    grads.append(p.grad)

        info = {
            "algo/value_loss": value_loss.item(),
            "algo/action_loss": action_loss.item(),
        }

        return grads, info


    def apply_gradients(self, gradients=None):
        """
        Take an optimization step, previously setting new gradients if provided.

        Parameters
        ----------
        gradients: list of tensors
            List of actor_critic gradients.
        """
        if gradients is not None:
            for g, p in zip(gradients, self.actor_critic.parameters()):
                if g is not None:
                    p.grad = torch.from_numpy(g).to(self.device)

        self.pi_optimizer.step()
        self.v_optimizer.step()
        self.iter += 1

    def set_weights(self, weights):
        """
        Update actor critic with the given weights.

        Parameters
        ----------
        weights: dict of tensors
            Dict containing actor_critic weights to be set.
        """
        self.actor_critic.load_state_dict(weights)
        self.iter += 1

    def update_algo_parameter(self, parameter_name, new_parameter_value):
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
        if parameter_name == "lr_v":
            for param_group in self.v_optimizer.param_groups:
                param_group['lr'] = new_parameter_value
        elif parameter_name == "lr_pi":
            for param_group in self.pi_optimizer.param_groups:
                param_group['lr'] = new_parameter_value
