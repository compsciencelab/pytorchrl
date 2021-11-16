from pytorchrl.agent.actors.planner.reward_functions import get_reward_function
from typing import Tuple
import gym
import numpy as np
import torch

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

    def _get_discrete_action(self, states: torch.Tensor)-> torch.Tensor:
        return torch.randint(self.action_space, size=(self.n_planner, self.action_space.n)).to(self.device)

    def _get_continuous_action(self, states: torch.Tensor)-> torch.Tensor:
        actions = np.random.uniform(low=self.action_low,
                                    high=self.action_high,
                                    size=(self.n_planner, self.action_space.shape[0]))
        return torch.from_numpy(actions).to(self.device)        


class MPC():
    def __init__(self, action_space, n_planner=50, planning_depth=50, device="cpu") -> None:
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
        
        self.n_planner = n_planner
        self.planning_depth = planning_depth
        self.device = device
        self.policy = RandomPolicy(action_space=action_space,
                                   action_type=self.action_type,
                                   action_high=self.action_high,
                                   action_low=self.action_low,
                                   n_planner=n_planner,
                                   device=device)
        
    def get_next_action(self, state: torch.Tensor, model: torch.nn.Module, noise: bool=False)-> torch.Tensor:

        states = state.repeat((self.n_planner, 1)).to(self.device)
        actions, returns = self.run_mpc(states, model)
        optimal_action = actions[returns.argmax()]
        if noise and self.action_type=="continuous":
            optimal_action += torch.normal(torch.zeros(optimal_action.shape),
                                           torch.ones(optimal_action.shape) * 0.005).to(self.device)
        return optimal_action
        
    def run_mpc(self, states: torch.Tensor, model: torch.nn.Module)-> Tuple[torch.Tensor, torch.Tensor]:
        
        returns = torch.zeros((self.n_planner, 1)).to(self.device)
        for i in range(self.planning_depth):
            actions = self.policy.get_action(states)
            #print("STATE: ", states)
            #print("ACTIONS", actions)
            with torch.no_grad():
                next_states, rewards = model.predict(states, actions)
            #print("NEXT_STATE: ", next_states)
            #time.sleep(10)
            states = next_states
            returns += rewards
            if i == 0:
                first_actions = actions
        return first_actions, returns
