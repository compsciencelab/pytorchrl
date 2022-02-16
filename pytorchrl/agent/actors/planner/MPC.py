from typing import Tuple
import gym
import numpy as np
import torch
import scipy.stats as stats

class MPC():
    """
    Base MPC class storing basic information for all MPC types.
    """
    def __init__(self, action_space, config, device)-> None:
        if type(action_space) == gym.spaces.discrete.Discrete:
            self.action_space = action_space.n
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
    """Random Shooting MPC class
    
    Parameters
    ----------
    action_space : gym.Space
        Environment action space.
    config: Namespace
        Configuration of the training run
    device: torch.device
        CPU or specific GPU where class computations will take place.
    """
    def __init__(self, action_space, config, device) -> None:
        super(RandomShooting, self).__init__(action_space=action_space, config=config, device=device)
        if self.action_type == "discrete":
            self.get_rollout_actions = self._get_discrete_actions
        elif self.action_type == "continuous":
            self.get_rollout_actions = self._get_continuous_actions
        else:
            raise ValueError("Selected action type does not exist!")


    def _get_discrete_actions(self, )-> torch.Tensor:
        """Samples random discrete actions"""
        return torch.randint(self.action_space, size=(self.n_planner, self.horizon, 1)).to(self.device)


    def _get_continuous_actions(self, )-> torch.Tensor:
        """Samples random continuous actions"""
        actions = np.random.uniform(low=self.action_low,
                                    high=self.action_high,
                                    size=(self.n_planner, self.horizon, self.action_space))
        return torch.from_numpy(actions).to(self.device).float()
    
    def get_action(self, state: torch.Tensor, model: torch.nn.Module, noise: bool=False)-> torch.Tensor:
        """Random shooting action planning process

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
        initial_states = state.repeat((self.n_planner, 1)).to(self.device)
        rollout_actions = self.get_rollout_actions()
        returns = self.compute_returns(initial_states, rollout_actions, model)
        optimal_action = rollout_actions[:, 0, :][returns.argmax()]

        if noise and self.action_type=="continuous":
            optimal_action += torch.normal(torch.zeros(optimal_action.shape),
                                           torch.ones(optimal_action.shape) * 0.005).to(self.device)
        return optimal_action


    def compute_returns(self, states: torch.Tensor, actions: torch.Tensor, model: torch.nn.Module)-> Tuple[torch.Tensor]:
        """Calculates the trajectory returns
        
        Parameters
        ----------
        states: torch.Tensor
            Trajectory states
        actions: torch.Tensor
            Trajectory actions
        model: dynamics Model
            Calculates the next states and rewards
        
        Returns
        -------
        returns: torch.Tensor
            Trajectory returns of the RS MPC
        
        """
        returns = torch.zeros((self.n_planner, 1)).to(self.device)
        for t in range(self.horizon):
            with torch.no_grad():
                states, rewards = model.predict(states, actions[:, t, :])
            returns += rewards

        return returns

class CEM(MPC):
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
    def __init__(self, action_space, config, device=None)-> None:
        super(CEM, self).__init__(action_space=action_space, config=config, device=device)

        self.iter_update_steps = config.iter_update_steps
        self.k_best = config.k_best
        self.update_alpha = config.update_alpha
        self.epsilon = 0.001
        self.device = device
        self.lb = -1
        self.ub = 1
        
    def get_action(self, state: torch.Tensor, model: torch.nn.Module, noise:bool=False):
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

        elite_actions = action_hist[idxs][-self.k_best:, :].squeeze(1) # sorted (elite, horizon x action_space)
        k_best_rewards = rewards[idxs][-self.k_best:, :].squeeze(-1)

        assert k_best_rewards.shape == (self.k_best, 1)
        assert elite_actions.shape == (self.k_best, self.horizon*self.action_space)
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
        assert best_actions.shape == (self.k_best, self.horizon*self.action_space)

        new_mu = best_actions.mean(0)
        new_var = best_actions.var(0)
        # Softupdate
        mu = (self.update_alpha * old_mu + (1.0 - self.update_alpha) * new_mu)
        var = (self.update_alpha * old_var + (1.0 - self.update_alpha) * new_var)
        assert mu.shape == (self.horizon*self.action_space, )
        assert var.shape == (self.horizon*self.action_space, )
        return mu, var


class PDDM(MPC):
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
    def __init__(self, action_space, config, device=None)-> None:
        super(PDDM, self).__init__(action_space=action_space, config=config, device=device)

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

        c = np.exp(self.gamma * (returns -np.max(returns)))
        d = np.sum(c) + 1e-10
        assert c.shape == (self.n_planner, 1)
        assert d.shape == (), "Has shape {}".format(d.shape)
        c_expanded = c[:, :, None]
        assert c_expanded.shape == (self.n_planner, 1, 1)
        weighted_actions = c_expanded * action_hist
        #print("D", d)
        #print("weighted_actions", weighted_actions.sum(0))
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
                actions[:, t, :] = self.beta * (self.mu[t, :] + u[:, t, :]) + (1 - self.beta) * actions[:, t-1, :]
        assert actions.shape == (self.n_planner, self.horizon, self.action_space), "Has shape {} but should have shape {}".format(actions.shape, (self.n_planner, self.horizon, self.action_space))
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