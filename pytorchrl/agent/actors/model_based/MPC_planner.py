import gym
import torch
from pytorchrl.agent.actors.base import Actor


class Planner:
    """
    Base Planner class storing basic information for all MPC types.
    """

    def __init__(self, world_model, action_space, n_planner, horizon, device) -> None:
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
            raise ValueError("Unknown action space")

        self.device = device
        self.n_planner = n_planner
        self.horizon = horizon
        self.world_model = world_model

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
            creates a new Planner class instance.
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

        return create_actor_instance

    def get_action(self, state: torch.Tensor, model: torch.nn.Module, noise: bool = False) -> torch.Tensor:
        raise NotImplementedError

