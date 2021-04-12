import torch
import numpy as np
from collections import deque
import pytorchrl as prl
from pytorchrl.agent.storages.off_policy.replay_buffer import ReplayBuffer as S


def dim0_reshape(tensor, size):
    """
    Reshapes tensor so indices are defined like this:

    00, 01, 02, 03, 04, 05, 06, 07, 08, 09, size + 1, ..., self.max_size
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, size + 1, ..., self.max_size
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, size + 1, ..., self.max_size

    """
    return np.moveaxis(tensor, [0, 1], [1, 0])[:, 0: size].reshape(-1, *tensor.shape[2:])


class NStepReplayBuffer(S):
    """
    Storage class for Off-Policy with multi step learning (https://arxiv.org/abs/1710.02298).

    Parameters
    ----------
    size : int
        Storage capacity along time axis.
    device : torch.device
        CPU or specific GPU where data tensors will be placed and class
        computations will take place. Should be the same device where the
        actor model is located.
    actor : Actor
        Actor class instance.
    algorithm : Algorithm
        Algorithm class instance
    n_step : int or float
        Number of future steps used to computed the truncated n-step return value.
    """

    # Data fields to store in buffer and contained in the generated batches
    storage_tensors = prl.DataTransitionKeys

    def __init__(self, size, device, actor, algorithm, n_step=1):

        algorithm._update_every += n_step - 1

        super(NStepReplayBuffer, self).__init__(
            size=size,  device=device, actor=actor, algorithm=algorithm)

        self.n_step = n_step
        self.gamma = algorithm.gamma
        self.n_step_buffer = {k: deque(maxlen=n_step) for k in self.storage_tensors}

    @classmethod
    def create_factory(cls, size, n_step=1):
        """
        Returns a function that creates NStepReplayBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.
        n_step : int or float
            Number of future steps used to computed the truncated n-step return value.

        Returns
        -------
        create_buffer_instance : func
            creates a new NStepReplayBuffer class instance.
        """
        def create_buffer(device, actor, algorithm):
            """Create and return a NStepReplayBuffer instance."""
            return cls(size, device, actor, algorithm, n_step)
        return create_buffer

    def insert_transition(self, sample):
        """
        Store new transition sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """

        # Data tensors lazy initialization
        if self.size == 0 and self.data[prl.OBS] is None:
            self.init_tensors(sample)

        # If using memory, save fixed length consecutive overlapping sequences
        if self.recurrent_actor and self.step % self.sequence_length == 0 and self.step != 0:
            next_seq_overlap = self.get_data_slice(self.step - self.overlap_length, self.step)
            self.insert_data_slice(next_seq_overlap)

        # Add obs, rhs, done, act and rew to n_step buffer
        self.n_step_buffer[prl.OBS].append(sample[prl.OBS])
        self.n_step_buffer[prl.REW].append(sample[prl.REW])
        self.n_step_buffer[prl.ACT].append(sample[prl.ACT])
        self.n_step_buffer[prl.RHS].append(sample[prl.RHS])
        self.n_step_buffer[prl.DONE].append(sample[prl.DONE])

        if len(self.n_step_buffer[prl.OBS]) == self.n_step:

            # Add obs2, rhs2 and done2 directly
            for k in (prl.OBS2, prl.RHS2, prl.DONE2):
                if isinstance(sample[k], dict):
                    for x, y in sample[k].items():
                        self.data[k][x][self.step] = y.cpu()
                else:
                    self.data[k][self.step] = sample[k].cpu()

            # Compute done and rew
            (self.data[prl.REW][self.step],
             self.data[prl.DONE][self.step]) = self._nstep_return()

            # Get obs, rhs and act from step buffer
            for k in (prl.OBS, prl.RHS, prl.ACT):
                tensor = self.n_step_buffer[k].popleft()
                if isinstance(tensor, dict):
                    for x, y in tensor.items():
                        self.data[k][x][self.step] = y.cpu()
                else:
                    self.data[k][self.step] = tensor.cpu()

            # Update
            self.step = (self.step + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def _nstep_return(self):
        """
        Computes truncated n-step returns.

        Returns
        -------
        ret : numpy.ndarray
            Next sample returns, to store in buffer.
        done : numpy.ndarray
            Next sample dones, to store in buffer.
        """

        ret = self.n_step_buffer[prl.REW][self.n_step - 1].clone()
        done = self.n_step_buffer[prl.DONE][self.n_step - 1].clone()
        for i in reversed(range(self.n_step - 1)):
            ret = ret * self.gamma * (1 - self.n_step_buffer[prl.DONE][i + 1])\
                  + self.n_step_buffer[prl.REW][i]
            done = done + self.n_step_buffer[prl.DONE][i]

        self.n_step_buffer[prl.REW].popleft()
        self.n_step_buffer[prl.DONE].popleft()

        return ret.cpu(), done.cpu()

    def generate_batches(self, num_mini_batch, mini_batch_size, num_epochs=1):
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

        Yields
        ------
        batch : dict
            Generated data batches.
        """

        num_proc = self.data[prl.DONE].shape[1]

        for _ in range(num_mini_batch):

            if self.recurrent_actor:  # Batches to feed a recurrent actor

                sequences_x_batch = mini_batch_size // self.sequence_length + 1

                assert self.size % self.sequence_length == 0, \
                    "Buffer does not contain an integer number of complete rollout sequences"

                # Define batch structure
                batch = {k: [] if not isinstance(self.data[k], dict) else
                    {x: [] for x in self.data[k]} for k in self.storage_tensors}

                # Randomly select sequences
                seq_idxs = np.random.randint(0, num_proc * int(
                    self.size / self.sequence_length), size=sequences_x_batch)

                # Get data indexes
                idxs = []
                for idx in seq_idxs:
                    idxs += range(idx * self.sequence_length, (idx + 1) * self.sequence_length)

                # Fill up batch with data
                for k, v in self.data.items():
                    # Only first recurrent state in each sequence needed
                    positions = seq_idxs * self.sequence_length if k.startswith(prl.RHS) else idxs
                    if isinstance(v, dict):
                        for x, y in v.items():
                            t = dim0_reshape(y, self.size)[positions]
                            batch[k][x] = torch.as_tensor(t, dtype=torch.float32).to(self.device)
                    else:
                        t = dim0_reshape(v, self.size)[positions]
                        batch[k] = torch.as_tensor(t, dtype=torch.float32).to(self.device)

                batch.update({"n_step": self.n_step})
                yield batch

            else:
                batch = {k: {} for k in self.storage_tensors}
                idxs = np.random.randint(0, num_proc * self.size, size=mini_batch_size)
                for k, v in self.data.items():
                    if isinstance(v, dict):
                        for x, y in v.items():
                            batch[k][x] = torch.as_tensor(y[0:self.size].reshape(
                                -1, *y.shape[2:])[idxs], dtype=torch.float32).to(self.device)
                    else:
                        batch[k] = torch.as_tensor(v[0:self.size].reshape(
                            -1, *v.shape[2:])[idxs], dtype=torch.float32).to(self.device)

                batch.update({"n_step": self.n_step})
                yield batch

    def update_storage_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of the algorithm, change its value
        to `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Attribute name
        new_parameter_value : int or float
            New value for `parameter_name`.
        """
        if hasattr(self, parameter_name):
            if parameter_name == "max_size" and self.recurrent_actor:
                new_parameter_value = (new_parameter_value // self.sequence_length) * self.sequence_length
                new_parameter_value *= 2
                new_parameter_value += self.n_step
            setattr(self, parameter_name, new_parameter_value)
