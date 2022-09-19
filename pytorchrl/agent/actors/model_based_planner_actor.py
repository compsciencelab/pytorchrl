import gym
import torch

from pytorchrl.agent.actors.base import Actor
from pytorchrl.agent.actors.world_models import WorldModel
from pytorchrl.agent.actors.world_models.utils import StandardScaler


class ModelBasedPlannerActor(Actor):
    """ Actor Planner class for MB agents. """

    def __init__(self,
                 device,
                 horizon,
                 n_planner,
                 input_space,
                 action_space,
                 algorithm_name,
                 checkpoint=None,
                 world_model_class=None,
                 world_model_kwargs={}):

        super(ModelBasedPlannerActor, self).__init__(
            device=device,
            checkpoint=checkpoint,
            input_space=input_space,
            action_space=action_space)

        if type(action_space) == gym.spaces.discrete.Discrete:
            self.action_dims = action_space.n
            self.action_type = "discrete"
            self.action_low = None
            self.action_high = None
        elif type(action_space) == gym.spaces.box.Box:
            self.action_dims = action_space.shape[0]
            self.action_type = "continuous"
            self.action_low = action_space.low
            self.action_high = action_space.high
        else:
            raise ValueError("Unknown action space")

        self.horizon = horizon
        self.n_planner = n_planner
        self.algorithm_name = algorithm_name

        # ----- World Model ---------------------------------------------------

        self.create_world_dynamics_model(world_model_class, world_model_kwargs)

    @classmethod
    def create_factory(
            cls,
            input_space,
            action_space,
            algorithm_name,
            horizon,
            n_planner,
            restart_model=None,
            world_model_class=None,
            world_model_kwargs={}
    ):
        """
        Returns a function that creates actor critic instances.

        Parameters
        ----------
        horizon : int
            The horizon of online planning.
        n_planner : int
            Number of parallel planned trajectories.
        input_space : gym.Space
            Environment observation space.
        action_space : gym.Space
            Environment action space.
        algorithm_name : str
            Name of the RL algorithm used for learning.
        restart_model : str
            Path to a previously trained Actor checkpoint to be loaded.
        world_model_class : class
            PyTorch nn.Module to approximate world dynamics.
        world_model_kwargs
            Keyword arguments for the world model class.

        Returns
        -------
        create_actor_instance : func
            creates a new OffPolicyActor class instance.
        """

        def create_actor_instance(device):
            """Create and return an actor critic instance."""
            policy = cls(device=device,
                         horizon=horizon,
                         n_planner=n_planner,
                         input_space=input_space,
                         action_space=action_space,
                         algorithm_name=algorithm_name,
                         checkpoint=restart_model,
                         world_model_class=world_model_class,
                         world_model_kwargs=world_model_kwargs)
            policy.to(device)

            try:
                policy.try_load_from_checkpoint()
            except RuntimeError:
                pass

            return policy

        return create_actor_instance

    @property
    def is_recurrent(self):
        """Returns True if the actor network are recurrent."""
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of policy recurrent hidden state"""
        return 1

    def actor_initial_states(self, obs):
        """
        Returns all actor inputs required to predict initial action.

        Parameters
        ----------
        obs : torch.tensor
            Initial environment observation.

        Returns
        -------
        obs : torch.tensor
            Initial environment observation.
        rhs : dict
            Initial recurrent hidden state (will contain zeroes).
        done : torch.tensor
            Initial done tensor, indicating the environment is not done.
        """

        if isinstance(obs, dict):
            num_proc = list(obs.values())[0].size(0)
            dev = list(obs.values())[0].device
        else:
            num_proc = obs.size(0)
            dev = obs.device

        done = torch.zeros(num_proc, 1).to(dev)

        try:
            rhs = self.policy_net.memory_net.get_initial_recurrent_state(num_proc).to(dev)
        except Exception:
            rhs = torch.zeros(num_proc, self.recurrent_hidden_state_size).to(dev)

        rhs = {"world_model_rhs": rhs}
        return obs, rhs, done

    def get_prediction(self, obs, act, rhs, done, deterministic=False):
        """
        Predict and return next action, along with other information.

        Parameters
        ----------
        obs : torch.tensor
            Current environment observation.
        act : torch.tensor
            Action to take given obs.
        rhs : dict
            Current recurrent hidden states.
        done : torch.tensor
            Current done tensor, indicating if episode has finished.
        deterministic : bool
            Whether to randomly sample action from predicted distribution or take the mode.

        Returns
        -------
        next_states : torch.Tensor
            Next states.
        rewards : torch.Tensor
            Reward prediction.
        """

        next_states, rewards = self.dynamics_model.predict(obs, act)

        return next_states, rewards

    def create_world_dynamics_model(self, world_model_class, world_model_kwargs):
        """
        Create a world model instance and define it as class attribute under the name `world_wodel`.

        Parameters
        ----------
        world_model_class : class
            WorldModel class
        world_model_kwargs : dict
            WorldModel class arguments
        """

        if world_model_class is None:
            world_model_class = WorldModel

        data_scaler = StandardScaler(self.device)
        dynamics_model = world_model_class(
            self.device, self.input_space, self.action_space,
            data_scaler, **world_model_kwargs).to(self.device)

        setattr(self, "dynamics_model", dynamics_model)
