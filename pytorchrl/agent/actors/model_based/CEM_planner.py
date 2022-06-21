import gym
import numpy as np
import torch
import scipy.stats as stats
from pytorchrl.agent.actors.model_based.base_planner import Planner


class CEMPlanner(Planner):
    """Cross Entropy Method MPC class

    Parameters
    ----------
    action_space : gym.Space
        Environment action space.
    config: Namespace
        Configuration of the training run
    device: torch.device
        CPU or specific GPU where class computations will take place.
    """

    def __init__(self, action_space, config, device=None) -> None:
        super(CEMPlanner, self).__init__(action_space=action_space, config=config, device=device)

        self.iter_update_steps = config.iter_update_steps
        self.k_best = config.k_best
        self.update_alpha = config.update_alpha
        self.epsilon = 0.001
        self.device = device
        self.lb = -1
        self.ub = 1

    def get_action(self, state: torch.Tensor, model: torch.nn.Module, noise: bool = False):
        """CEM action planning process

        Parameters
        ----------
        state: torch.Tensor
            Current state in the environment which is used to start the trajectories for planning
        model: torch.nn.Module
            Dynamics model
        noise: bool
            Adding noise to the optimal action if set to True

        Returns
        -------
        action: torch.Tensor:
            First action in the trajectory that achieves the highest reward
        """
        initial_state = state.repeat(self.n_planner, 1).to(self.device)
        mu = np.zeros(self.horizon * self.action_space)
        var = 5 * np.ones(self.horizon * self.action_space)
        X = stats.truncnorm(self.lb, self.ub, loc=np.zeros_like(mu), scale=np.ones_like(mu))
        i = 0
        while ((i < self.iter_update_steps) and (np.max(var) > self.epsilon)):
            states = initial_state
            returns = np.zeros((self.n_planner, 1))
            # variables
            lb_dist = mu - self.lb
            ub_dist = self.ub - mu
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            actions = X.rvs(size=[self.n_planner, self.horizon * self.action_space]) * np.sqrt(constrained_var) + mu
            actions = np.clip(actions, self.lb, self.ub)
            actions_t = torch.from_numpy(actions.reshape(self.n_planner,
                                                         self.horizon,
                                                         self.action_space)).float().to(self.device)
            for t in range(self.horizon):
                with torch.no_grad():
                    states, rewards = model.predict(states, actions_t[:, t, :])
                returns += rewards.cpu().numpy()

            k_best_rewards, k_best_actions = self.select_k_best(returns, actions)
            mu, var = self.update_gaussians(mu, var, k_best_actions)
            i += 1

        best_action_sequence = mu.reshape(self.horizon, -1)
        best_action = np.copy(best_action_sequence[0])
        assert best_action.shape == (self.action_space,)
        return torch.from_numpy(best_action).float().to(self.device)

    def select_k_best(self, rewards, action_hist):
        """Selects k action trajectories that led to the highest reward.

        Parameters
        ----------
        rewards: np.array
            Rewards per rollout
        action_history: np.array
            Action history for all rollouts

        Returns
        -------
        k_best_rewards: np.array
            K-rewards of the action trajectories that the highest reward value
        elite_actions: np.array
            Best action histories
        """
        assert rewards.shape == (self.n_planner, 1)
        idxs = np.argsort(rewards, axis=0)

        elite_actions = action_hist[idxs][-self.k_best:, :].squeeze(1)  # sorted (elite, horizon x action_space)
        k_best_rewards = rewards[idxs][-self.k_best:, :].squeeze(-1)

        assert k_best_rewards.shape == (self.k_best, 1)
        assert elite_actions.shape == (self.k_best, self.horizon * self.action_space)
        return k_best_rewards, elite_actions

    def update_gaussians(self, old_mu, old_var, best_actions):
        """Updates the mu and var value for the gaussian action sampling method.

        Parameters
        ----------
        old_mu: np.array
            Old mean value
        old_var: np.array
            Old variance value
        best_actions: np.array
            Action history that led to the highest reward

        Returns
        -------
        mu: np.array
            Updated mean values
        var: np.array
            Updated variance values

        """
        assert best_actions.shape == (self.k_best, self.horizon * self.action_space)

        new_mu = best_actions.mean(0)
        new_var = best_actions.var(0)

        # Softupdate
        mu = (self.update_alpha * old_mu + (1.0 - self.update_alpha) * new_mu)
        var = (self.update_alpha * old_var + (1.0 - self.update_alpha) * new_var)
        assert mu.shape == (self.horizon * self.action_space,)
        assert var.shape == (self.horizon * self.action_space,)
        return mu, var

