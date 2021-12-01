import torch
import pytorchrl as prl
from pytorchrl.agent.storages.on_policy.vanilla_on_policy_buffer import VanillaOnPolicyBuffer as B


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
    envs : VecEnv
        Vector of environments instance.
    actor : Actor
        Actor class instance.
    algorithm : Algorithm
        Algorithm class instance.
    """

    # Data fields to store in buffer and contained in generated batches
    storage_tensors = prl.OnPolicyDataKeys

    def __init__(self, size, device, actor, algorithm, envs, gae_lambda=0.95):

        super(GAEBuffer, self).__init__(
            size=size,
            envs=envs,
            device=device,
            actor=actor,
            algorithm=algorithm)

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
        def create_buffer_instance(device, actor, algorithm, envs):
            """Create and return a OnPolicyGAEBuffer instance."""
            return cls(size, device, actor, algorithm, envs, gae_lambda)
        return create_buffer_instance

    @property
    def used_capacity(self):
        """Returns the step up to which storage is full with env transitions."""
        return self.step - 1 if self.step != 0 else self.max_size

    def compute_returns(self, rewards, returns, values, dones, gamma):
        """Compute return values."""
        gae_lambda, length, gae = self.gae_lambda, self.used_capacity, 0
        returns[length].copy_(values[length])
        for step in reversed(range(length)):
            delta = (rewards[step] + gamma * values[step + 1] * (1.0 - dones[step + 1]) - values[step])
            gae = delta + gamma * gae_lambda * (1.0 - dones[step + 1]) * gae
            returns[step] = gae + values[step]


