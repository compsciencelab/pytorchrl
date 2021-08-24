from agent.actors.planner.reward_functions import get_reward_function
import gym
import numpy as np
import torch
from copy import deepcopy

class RandomPolicy():
    def __init__(self, action_space, action_type, action_high, action_low, n_planner, device=None) -> None:
        self.n_planner = n_planner
        self.device = device
        self.action_space = action_space
        self.action_high = action_high
        self.action_low = action_low
        if action_type == "discrete":
            self.get_action = self._get_discrete_action
        elif action_type == "continuous":
            self.get_action = self._get_continuous_action
        else:
            raise ValueError("Selected action type does not exist!")

    def _get_discrete_action(self)-> torch.Tensor:
        return torch.randint(self.action_space, size=(self.n_planner, self.action_space)).to(self.device)

    def _get_continuous_action(self)-> torch.Tensor:
        actions = np.random.uniform(low=self.action_low,
                                    high=self.action_high,
                                    size=(self.n_planner, self.action_space))
        return torch.from_numpy(actions).to(self.device)        


class MPC():
    def __init__(self, env_name, action_space, n_planner=50, depth=50, device="cpu") -> None:
        if type(action_space) == gym.spaces.discrete.Discrete:
            self.action_space = 1
            self.action_type = "discrete"
        elif type(action_space) == gym.spaces.box.Box:
            self.action_space = action_space.shape[0]
            self.action_type = "continuous"
            self.action_low = action_space.low
            self.action_high = action_space.high
        else:
            raise ValueError ("Unknonw action space")
        
        self.n_planner = n_planner
        self.depth = depth
        self.device = device
        self.policy = RandomPolicy(action_space=action_space,
                                   action_type=self.action_type,
                                   action_high=self.action_high,
                                   action_low=self.action_low,
                                   n_planner=n_planner,
                                   device=device)
        
        self.reward_function = get_reward_function(env_name)

    def get_next_action(self, state, model, noise=False):

        states = torch.from_numpy(state).repeat((self.n_planner, 1)).to(self.device)
        actions, returns = self.run_mpc(states, model)
        optimal_action = actions[returns.argmax()].cpu().numpy()
        if noise and self.action_type=="continuous":
            optimal_action += np.random.normal(0, 0.005, size=optimal_action.shape)
        return optimal_action
        
    def run_mpc(self, states, model):
        
        returns = torch.zeros((self.n_planner, 1))
        for i in range(self.depth):
            actions = self.policy.get_actions(states)
            with torch.no_grad():
                next_states = model.get_next_state(states, actions)
            rewards = self.reward_function(states, actions)
            states = deepcopy(next_states) # Does it need deepcopy?
            returns += rewards.cpu()
            if i == 0:
                first_actions = deepcopy(actions) # Does it need deepcopy?


        return first_actions, returns
