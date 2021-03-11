import torch
import numpy as np
from pytorchrl_extensions.storages.ere_buffer import EREBuffer as B


class HERBuffer(B):
    """
    Storage class for Off-Policy algorithms using HER (https://arxiv.org/abs/1707.01495).

    Parameters
    ----------
    size : int
        Storage capacity along time axis.
    device: torch.device
        CPU or specific GPU where data tensors will be placed and class
        computations will take place. Should be the same device where the
        actor model is located.
    her_function : func
        Function to update obs, rhs, obs2 and rew according to HER paper.

    Warnings
    --------
    When using an environment vector of size larger than 1, episode sized must
    be of a fixed length. This HER implementation is not able to deal with envs
    of variable episode length, except in the case of environment vector size 1.

    """

    # Accepted data fields. Inserting other fields will raise AssertionError
    off_policy_data_fields = ("obs", "obs2", "rhs", "act", "rew", "done")

    def __init__(self, size, device, her_function, eta=1.0, cmin=5000,
                 epsilon=0.0, alpha=0.0, beta=1.0, gamma=0.99, n_step=1):

        super(HERBuffer, self).__init__(
            size=size, device=device, gamma=gamma, n_step=n_step,
            epsilon=epsilon, alpha=alpha, beta=beta, eta=eta, cmin=cmin)

        self.her_function = her_function
        self.last_episode_start = 0

    @classmethod
    def create_factory(cls, size, her_function=lambda o, rhs, o2, r, initial_state, final_state : (o, rhs, o2, r),
                       eta=1.0, cmin=5000, epsilon=0.0, alpha=0.0, beta=1.0, gamma=0.99, n_step=1):
        """
        Returns a function that creates HERBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.

        Returns
        -------
        create_buffer_instance : func
            creates a new HERBuffer class instance.
        """

        def create_buffer(device):
            """Create and return a HERBuffer instance."""
            return cls(size, device, her_function, eta, cmin, epsilon, alpha, beta, gamma, n_step)

        return create_buffer

    def insert(self, sample):
        """
        Store new transition sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """
        if self.size == 0 and self.data["obs"] is None:  # data tensors lazy initialization
            self.init_tensors(sample)

        # Add obs2 directly
        self.data["obs2"][self.step] = sample["obs2"].cpu()

        # Add obs, rew, rhs, done and act to n_step buffer
        self.n_step_buffer["obs"].append(sample["obs"].cpu())
        self.n_step_buffer["rew"].append(sample["rew"].cpu())
        self.n_step_buffer["act"].append(sample["act"].cpu())
        self.n_step_buffer["rhs"].append(sample["rhs"].cpu())
        self.n_step_buffer["done"].append(sample["done"].cpu())

        if len(self.n_step_buffer["obs"]) == self.n_step:
            self.data["rew"][self.step], self.data["done"][self.step] = self._nstep_return()
            self.data["obs"][self.step] = self.n_step_buffer["obs"].popleft()
            self.data["act"][self.step] = self.n_step_buffer["act"].popleft()
            self.data["rhs"][self.step] = self.n_step_buffer["rhs"].popleft()

        self.step = (self.step + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        # Handle end of episode - only works with fixed episode length case !!
        if (self.data["done"][self.step - 1] == 1.0).all():
            self.handle_end_of_episode()

    def handle_end_of_episode(self):
        """ _ """

        final_state = np.copy(self.data["obs"][self.step - 1])
        initial_state = np.copy(self.data["obs"][self.last_episode_start])

        # could also be self.step. I am choosing not to add the transition
        # in which obs and obs2 belong to different episodes
        current_step = self.step - 1

        for i in range(self.last_episode_start, current_step):

            obs, rhs, obs2, rew = self.her_function(
                np.copy(self.data["obs"][i]),
                np.copy(self.data["rhs"][i]),
                np.copy(self.data["obs2"][i]),
                np.copy(self.data["rew"][i]),
                initial_state,
                final_state)

            sample = {
                "obs": torch.tensor(obs), "rhs": torch.tensor(rhs),
                "rew": torch.tensor(rew), "obs2": torch.tensor(obs2),
                "act": torch.tensor(np.copy(self.data["act"][i])),
                "done": torch.tensor(np.copy(self.data["done"][i])),
            }

            # Add obs2 directly
            self.data["obs2"][self.step] = sample["obs2"]

            # Add obs, rew, rhs, done and act to n_step buffer
            self.n_step_buffer["obs"].append(sample["obs"])
            self.n_step_buffer["rew"].append(sample["rew"])
            self.n_step_buffer["act"].append(sample["act"])
            self.n_step_buffer["rhs"].append(sample["rhs"])
            self.n_step_buffer["done"].append(sample["done"])

            if len(self.n_step_buffer["obs"]) == self.n_step:
                self.data["rew"][self.step], self.data["done"][self.step] = self._nstep_return()
                self.data["obs"][self.step] = self.n_step_buffer["obs"].popleft()
                self.data["act"][self.step] = self.n_step_buffer["act"].popleft()
                self.data["rhs"][self.step] = self.n_step_buffer["rhs"].popleft()

            self.step = (self.step + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

        self.last_episode_start = self.step

