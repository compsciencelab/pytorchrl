import torch
import numpy as np
from collections import deque
from pytorchrl.agent.storages.off_policy.replay_buffer import ReplayBuffer as S


class NStepReplayBuffer(S):
    """
    Storage class for Off-Policy with multi step learning (https://arxiv.org/abs/1710.02298).

    Parameters
    ----------
    size : int
        Storage capacity along time axis.
    gamma: float
        Discount factor.
    n_step: int or float
        Number of future steps used to computed the truncated n-step return value.
    device: torch.device
        CPU or specific GPU where data tensors will be placed and class
        computations will take place. Should be the same device where the
        actor model is located.
    """

    # Accepted data fields. Inserting other fields will raise AssertionError
    off_policy_data_fields = ("obs", "obs2", "rhs", "act", "rew", "done")

    def __init__(self, size, gamma, n_step, device):

        super(NStepReplayBuffer, self).__init__(size=size,  device=device)

        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = {k: deque(maxlen=n_step) for k in self.off_policy_data_fields}

    @classmethod
    def create_factory(cls, size, gamma, n_step=1):
        """
        Returns a function that creates NStepReplayBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.
        gamma: float
            Discount factor.
        n_step: int or float
            Number of future steps used to computed the truncated n-step return value.

        Returns
        -------
        create_buffer_instance : func
            creates a new NStepReplayBuffer class instance.
        """
        def create_buffer(device):
            """Create and return a NStepReplayBuffer instance."""
            return cls(size, gamma, n_step, device)
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

    def _nstep_return(self):
        """
        Computes truncated n-step returns.

        Returns
        -------
        ret: numpy.ndarray
            Next sample returns, to store in buffer.
        done: numpy.ndarray
            Next sample dones, to store in buffer.
        """

        ret = self.n_step_buffer["rew"][self.n_step - 1].clone()
        done = self.n_step_buffer["done"][self.n_step - 1].clone()
        for i in reversed(range(self.n_step - 1)):
            ret = ret * self.gamma * (1 - self.n_step_buffer["done"][i + 1]) + self.n_step_buffer["rew"][i]
            done = done + self.n_step_buffer["done"][i]

        self.n_step_buffer["rew"].popleft()
        self.n_step_buffer["done"].popleft()

        return ret.cpu(), done.cpu()

    def generate_batches(self, num_mini_batch, mini_batch_size, num_epochs=1, recurrent_ac=False):
        """
        Returns a batch iterator to update actor.

        Parameters
        ----------
        num_mini_batch : int
           Number mini batches per epoch.
        mini_batch_size : int
            Number of samples contained in each mini batch.
        num_epochs : int
            Number of epochs.
        recurrent_ac : bool
            Whether actor policy is a RNN or not.
        shuffle : bool
            Whether to shuffle collected data or generate sequential

        Yields
        ______
        batch : dict
            Generated data batches. Contains also n_step information.
        """
        num_proc = self.data["obs"].shape[1]

        if recurrent_ac:  # Batches to a feed recurrent actor
            raise NotImplementedError

        else:  # Batches for a feed forward actor
            for _ in range(num_mini_batch):
                idxs = np.random.randint(0, num_proc * self.size, size=mini_batch_size)
                batch = dict(
                    obs=self.data["obs"][0:self.size].reshape(-1, *self.data["obs"].shape[2:])[idxs],
                    rhs=self.data["rhs"][0:self.size].reshape(-1, *self.data["rhs"].shape[2:])[idxs],
                    act=self.data["act"][0:self.size].reshape(-1, *self.data["act"].shape[2:])[idxs],
                    rew=self.data["rew"][0:self.size].reshape(-1, *self.data["rew"].shape[2:])[idxs],
                    obs2=self.data["obs2"][0:self.size].reshape(-1, *self.data["obs2"].shape[2:])[idxs],
                    done=self.data["done"][0:self.size].reshape(-1, *self.data["done"].shape[2:])[idxs])
                batch = {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in batch.items()}
                batch.update({"n_step": self.n_step})
                yield batch