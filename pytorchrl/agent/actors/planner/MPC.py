from typing import Tuple
import gym
import numpy as np
import torch
import scipy.stats as stats

class MPC():
    def __init__(self, action_space, config, device)-> None:
        if type(action_space) == gym.spaces.discrete.Discrete:
            self.action_space = 1
            self.action_type = "discrete"
            self.action_low = None
            self.action_high = None
        elif type(action_space) == gym.spaces.box.Box:
            self.action_space = action_space.shape[0]
            self.action_type = "continuous"
            self.action_low = action_space.low
            self.action_high = action_space.high
        else:
            raise ValueError ("Unknonw action space")
        
        self.device = device
        self.n_planner = config.n_planner
        self.horizon = config.horizon
    
    def get_action(self, state: torch.Tensor, model: torch.nn.Module, noise: bool=False)-> torch.Tensor:
        raise NotImplementedError


class RandomShooting(MPC):
    def __init__(self, action_space, config, device) -> None:
        super(RandomShooting, self).__init__(action_space=action_space, config=config, device=device)
        if self.action_type == "discrete":
            self.get_rollout_actions = self._get_discrete_actions
        elif self.action_type == "continuous":
            self.get_rollout_actions = self._get_continuous_actions
        else:
            raise ValueError("Selected action type does not exist!")


    def _get_discrete_actions(self, )-> torch.Tensor:
        return torch.randint(self.action_space, size=(self.n_planner, self.horizon, 1)).to(self.device)


    def _get_continuous_actions(self, )-> torch.Tensor:
        actions = np.random.uniform(low=self.action_low,
                                    high=self.action_high,
                                    size=(self.n_planner, self.horizon, self.action_space))
        return torch.from_numpy(actions).to(self.device)
    
    def get_action(self, state: torch.Tensor, model: torch.nn.Module, noise: bool=False)-> torch.Tensor:

        initial_states = state.repeat((self.n_planner, 1)).to(self.device)
        rollout_actions = self.get_rollout_actions()
        returns = self.compute_returns(initial_states, rollout_actions, model)
        optimal_action = rollout_actions[:, 0, :][returns.argmax()]

        if noise and self.action_type=="continuous":
            optimal_action += torch.normal(torch.zeros(optimal_action.shape),
                                           torch.ones(optimal_action.shape) * 0.005).to(self.device)
        return optimal_action


    def compute_returns(self, states: torch.Tensor, actions: torch.Tensor, model: torch.nn.Module)-> Tuple[torch.Tensor, torch.Tensor]:
        
        returns = torch.zeros((self.n_planner, 1)).to(self.device)
        for t in range(self.horizon):
            with torch.no_grad():
                states, rewards = model.predict(states, actions[:, t, :])
            returns += rewards

        return returns

class CEM(MPC):
    def __init__(self, action_space, config, device=None)-> None:
        super(CEM, self).__init__(action_space=action_space, config=config, device=device)

        self.iter_update_steps = config.iter_update_steps
        self.k_best = config.k_best
        self.update_alpha = config.update_alpha # Add this to CEM config
        self.epsilon = 0.001
        self.ub = 1
        self.lb = -1
        self.device = device
        
    def get_action(self, state, model, noise=False):
        initial_state = state.repeat(self.n_planner, 1).to(self.device)
        mu = np.zeros(self.horizon*self.action_space)
        var = 5 * np.ones(self.horizon*self.action_space)
        X = stats.truncnorm(self.lb, self.ub, loc=np.zeros_like(mu), scale=np.ones_like(mu))
        i = 0
        while ((i < self.iter_update_steps) and (np.max(var) > self.epsilon)):
            states = initial_state
            returns = np.zeros((self.n_planner, 1))
            #variables
            lb_dist = mu - self.lb
            ub_dist = self.ub - mu
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            
            actions = X.rvs(size=[self.n_planner, self.horizon*self.action_space]) * np.sqrt(constrained_var) + mu
            actions_t = torch.from_numpy(np.clip(actions, -1, 1).reshape(self.n_planner,
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
        best_action = np.copy(best_action_sequence[-1])
        assert best_action.shape == (self.action_space,)
        return torch.from_numpy(best_action).float().to(self.device)
            
    
    def select_k_best(self, rewards, action_hist):
        assert rewards.shape == (self.n_planner, 1)
        idxs = np.argsort(rewards, axis=0)

        elite_actions = action_hist[idxs][-self.k_best:, :].squeeze(1) # sorted (elite, horizon x action_space)
        k_best_rewards = rewards[idxs][-self.k_best:, :].squeeze(-1)

        assert k_best_rewards.shape == (self.k_best, 1)
        assert elite_actions.shape == (self.k_best, self.horizon*self.action_space)
        return k_best_rewards, elite_actions


    def update_gaussians(self, old_mu, old_var, best_actions):
        assert best_actions.shape == (self.k_best, self.horizon*self.action_space)

        new_mu = best_actions.mean(0)
        new_var = best_actions.var(0)

        mu = (self.update_alpha * old_mu + (1.0 - self.update_alpha) * new_mu)
        var = (self.update_alpha * old_var + (1.0 - self.update_alpha) * new_var)
        assert mu.shape == (self.horizon*self.action_space, )
        assert var.shape == (self.horizon*self.action_space, )
        return mu, var


class PDDM(MPC):
    def __init__(self, action_space, config, device=None)-> None:
        super(PDDM, self).__init__(action_space=action_space, config=config, device=device)

        self.gamma = config.gamma
        self.beta = config.beta
        self.mu = np.zeros((self.horizon, self.action_space))
        self.device = device
        
    def get_action(self, state, model, noise=False):
        initial_states = state.repeat(self.n_planner, 1).to(self.device)
        actions, returns = self.get_pred_trajectories(initial_states, model)
        optimal_action = self.update_mu(actions, returns)
       
        if noise:
            optimal_action += np.random.normal(0, 0.005, size=optimal_action.shape)
        return torch.from_numpy(optimal_action).float().to(self.device)
        
    def update_mu(self, action_hist, returns):
        assert action_hist.shape == (self.n_planner, self.horizon, self.action_space)
        assert returns.shape == (self.n_planner, 1)

        c = np.exp(self.gamma * (returns) -np.max(returns))
        d = np.sum(c) + 1e-10
        assert c.shape == (self.n_planner, 1)
        assert d.shape == (), "Has shape {}".format(d.shape)
        c_expanded = c[:, :, None]
        assert c_expanded.shape == (self.n_planner, 1, 1)
        weighted_actions = c_expanded * action_hist
        self.mu = weighted_actions.sum(0) / d
        assert self.mu.shape == (self.horizon, self.action_space)       
        
        return self.mu[0]
    
    def sample_actions(self, past_action):
        u = np.random.normal(loc=0, scale=1.0, size=(self.n_planner, self.horizon, self.action_space))
        actions = u.copy()
        for t in range(self.horizon):
            if t == 0:
                actions[:, t, :] = self.beta * (self.mu[t, :] + u[:, t, :]) + (1 - self.beta) * past_action
            else:
                actions[:, t, :] = self.beta * (self.mu[t, :] + u[:, t, :]) + (1 - self.beta) * actions[:, t-1, :]
        assert actions.shape == (self.n_planner, self.horizon, self.action_space), "Has shape {} but should have shape {}".format(actions.shape, (self.n_planner, self.horizon, self.action_space))
        actions = np.clip(actions, -1, 1)
        return actions
    
    def get_pred_trajectories(self, states, model): 
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