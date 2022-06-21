import gym
import numpy as np
import torch
from pytorchrl.agent.actors.model_based.base_planner import Planner


class PDDMPlanner(Planner):
    """Filtering and Reward-Weighted Refinement (PDDM) MPC class

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
        super(PDDMPlanner, self).__init__(action_space=action_space, config=config, device=device)

        self.gamma = config.gamma
        self.beta = config.beta
        self.mu = np.zeros((self.horizon, self.action_space))
        self.device = device

    def get_action(self, state, model, noise=False):
        """PDDM action planning process

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
        initial_states = state.repeat(self.n_planner, 1).to(self.device)
        actions, returns = self.get_pred_trajectories(initial_states, model)
        optimal_action = self.update_mu(actions, returns)

        if noise:
            optimal_action += np.random.normal(0, 0.005, size=optimal_action.shape)
        return torch.from_numpy(optimal_action).float().to(self.device)

    def update_mu(self, action_hist, returns):
        """Updates the mean value for the action sampling distribution.

        Parameters
        ----------
        action_hist: np.array
            Action history of the planned trajectories.
        returns: np.array
            Returns of the planned trajectories.

        Returns
        -------
        mu: np.array
            Updates mean value.
        """
        assert action_hist.shape == (self.n_planner, self.horizon, self.action_space)
        assert returns.shape == (self.n_planner, 1)

        c = np.exp(self.gamma * (returns - np.max(returns)))
        d = np.sum(c) + 1e-10
        assert c.shape == (self.n_planner, 1)
        assert d.shape == (), "Has shape {}".format(d.shape)
        c_expanded = c[:, :, None]
        assert c_expanded.shape == (self.n_planner, 1, 1)
        weighted_actions = c_expanded * action_hist
        # print("D", d)
        # print("weighted_actions", weighted_actions.sum(0))
        self.mu = weighted_actions.sum(0) / d
        assert self.mu.shape == (self.horizon, self.action_space)

        return self.mu[0]

    def sample_actions(self, past_action):
        """Samples action trajectories.

        Parameters
        ----------
        past_action: np.array
            Previous action mean value.

        Returns
        -------
        actions: np.array
            Sampled action trajectories.
        """
        u = np.random.normal(loc=0, scale=1.0, size=(self.n_planner, self.horizon, self.action_space))
        actions = u.copy()
        for t in range(self.horizon):
            if t == 0:
                actions[:, t, :] = self.beta * (self.mu[t, :] + u[:, t, :]) + (1 - self.beta) * past_action
            else:
                actions[:, t, :] = self.beta * (self.mu[t, :] + u[:, t, :]) + (1 - self.beta) * actions[:, t - 1, :]
        assert actions.shape == (
        self.n_planner, self.horizon, self.action_space), "Has shape {} but should have shape {}".format(actions.shape,
                                                                                                         (
                                                                                                         self.n_planner,
                                                                                                         self.horizon,
                                                                                                         self.action_space))
        actions = np.clip(actions, self.action_low, self.action_high)
        return actions

    def get_pred_trajectories(self, states, model):
        """Calculates the returns when planning given a state and a model.

        Parameters
        ----------
        states: torch.Tensor
            Initial states that are used for the planning.
        model: dynamics model nn.Module
            The dynamics model that is used to predict the next state and reward.

        Returns
        -------
        actions: np.array
            Action history of the sampled trajectories used for planning.
        returns: np.array
            Returns of the action trajectories.
        """
        returns = np.zeros((self.n_planner, 1))
        np.random.seed()
        past_action = self.mu[0].copy()
        actions = self.sample_actions(past_action)
        torch_actions = torch.from_numpy(actions).float().to(self.device)
        for t in range(self.horizon):
            with torch.no_grad():
                actions_t = torch_actions[:, t, :]
                assert actions_t.shape == (self.n_planner, self.action_space)
                states, rewards = model.predict(states, actions_t)
            returns += rewards.cpu().numpy()
        return actions, returns
