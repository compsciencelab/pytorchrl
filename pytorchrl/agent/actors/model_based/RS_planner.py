from typing import Tuple
import gym
import numpy as np
import torch
from pytorchrl.agent.actors.model_based.base_planner import Planner


class RandomShootingPlanner(Planner):
    """Random Shooting MPC class

    Parameters
    ----------
    world_model : nn.Module
        World dynamics predictor
    action_space : gym.Space
        Environment action space.
    n_planner :

    horizon :

    device: torch.device
        CPU or specific GPU where class computations will take place.
    """

    def __init__(self, world_model, action_space, n_planner, horizon, device) -> None:
        super(RandomShootingPlanner, self).__init__(
            world_model=world_model, action_space=action_space,
            n_planner=n_planner, horizon=horizon, device=device)

        if self.action_type == "discrete":
            self.get_rollout_actions = self._get_discrete_actions
        elif self.action_type == "continuous":
            self.get_rollout_actions = self._get_continuous_actions
        else:
            raise ValueError("Selected action type does not exist!")

    @classmethod
    def create_factory(cls, world_model_factory, action_space, n_planner, horizon):
        """
        Returns a function that creates actor critic instances.

        Parameters
        ----------
        world_model_factory : func
            A function that creates a world model.
        action_space : gym.Space
            Environment action space.
        n_planner :

        horizon :


        Returns
        -------
        create_actor_instance : func
            creates a new RSPlanner class instance.
        """

        def create_actor_instance(device):
            """Create and return an actor critic instance."""
            actor = cls(world_model=world_model_factory(device),
                        device=device,
                        action_space=action_space,
                        n_planner=n_planner,
                        horizon=horizon)
            actor.to(device)
            return actor

    def _get_discrete_actions(self, ) -> torch.Tensor:
        """Samples random discrete actions"""
        return torch.randint(self.action_space, size=(self.n_planner, self.horizon, 1)).to(self.device)

    def _get_continuous_actions(self, ) -> torch.Tensor:
        """Samples random continuous actions"""
        actions = np.random.uniform(low=self.action_low,
                                    high=self.action_high,
                                    size=(self.n_planner, self.horizon, self.action_space))
        return torch.from_numpy(actions).to(self.device).float()

    def get_action(self, state: torch.Tensor, model: torch.nn.Module, noise: bool = False) -> torch.Tensor:
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

        if noise and self.action_type == "continuous":
            optimal_action += torch.normal(
                torch.zeros(optimal_action.shape),
                torch.ones(optimal_action.shape) * 0.005).to(self.device)

        return optimal_action

    def compute_returns(self, states: torch.Tensor, actions: torch.Tensor, model: torch.nn.Module) -> Tuple[
        torch.Tensor]:
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
