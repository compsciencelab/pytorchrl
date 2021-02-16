import torch
from .vanilla_on_policy_buffer import VanillaOnPolicyBuffer as B


class GAEBuffer(B):
    """
    Storage class for On-Policy algorithms with Generalized Advantage
    Estimator (GAE). https://arxiv.org/abs/1506.02438

    Parameters
    ----------
    size : int
        Storage capacity along time axis.
    gae_lambda : float
        GAE lambda parameter.
    device: torch.device
        CPU or specific GPU where data tensors will be placed and class
        computations will take place. Should be the same device where the
        actor model is located.

    Attributes
    ----------
    max_size : int
        Storage capacity along time axis.
    device: torch.device
        CPU or specific GPU where data tensors will be placed and class
        computations will take place.
    gae_lambda : float
        GAE lambda parameter.
    """

    # Accepted data fields. Inserting other fields will raise AssertionError
    on_policy_data_fields = ("obs", "obs2", "rhs", "act", "rew", "val", "logp", "done")

    def __init__(self, size, gae_lambda=0.95, device=torch.device("cpu")):

        super(GAEBuffer, self).__init__(
            size=size,
            device=device)

        self.gae_lambda = gae_lambda

    @classmethod
    def create_factory(cls, size, gae_lambda=0.95):
        """
        Returns a function that creates OnPolicyGAEBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.
        gae_lambda : float
            GAE lambda parameter.

        Returns
        -------
        create_buffer_instance : func
            creates a new OnPolicyBuffer class instance.
        """
        def create_buffer_instance(device):
            """Create and return a OnPolicyGAEBuffer instance."""
            return cls(size, gae_lambda, device)
        return create_buffer_instance

    def before_gradients(self, actor, algo):
        """
        Before updating actor policy model, compute returns and advantages.

        Parameters
        ----------
        actor : ActorCritic
            An actor class instance.
        algo : an algorithm class
            An algorithm class instance.
        """
        with torch.no_grad():
            _ = actor.get_action(
                self.data["obs"][self.step - 1],
                self.data["rhs"][self.step - 1],
                self.data["done"][self.step - 1])
            next_value = actor.get_value(
                self.data["obs"][self.step - 1],
                self.data["rhs"][self.step - 1],
                self.data["done"][self.step - 1]
            )

        self.data["val"][self.step] = next_value
        self.compute_returns(algo.gamma)
        self.compute_advantages()

    def after_gradients(self, actor, algo, batch, info):
        """
        After updating actor policy model, make sure self.step is at 0.

        Parameters
        ----------
        actor : Actor class
            An actor class instance.
        algo : Algo class
            An algorithm class instance.
        batch : dict
            Data batch used to compute the gradients.
        info : dict
            Additional relevant info from gradient computation.
        """
        self.data["obs"][0].copy_(self.data["obs"][self.step - 1])
        self.data["rhs"][0].copy_(self.data["rhs"][self.step - 1])
        self.data["done"][0].copy_(self.data["done"][self.step - 1])

        if self.step != 0:
            self.step = 0

    def compute_returns(self, gamma):
        """
        Compute return values.

        Parameters
        ----------
        gamma : float
            Algorithm discount factor parameter.
        """
        len = self.step if self.step != 0 else self.max_size
        gae = 0
        for step in reversed(range(len)):
            delta = (self.data["rew"][step] + gamma * self.data["val"][step + 1] * (
                1.0 - self.data["done"][step + 1]) - self.data["val"][step])
            gae = delta + gamma * self.gae_lambda * (1.0 - self.data["done"][step + 1]) * gae
            self.data["ret"][step] = gae + self.data["val"][step]
