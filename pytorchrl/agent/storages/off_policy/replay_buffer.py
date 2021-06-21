import numpy as np
import torch

import pytorchrl as prl
from pytorchrl.agent.storages.base import Storage as S


def dim0_reshape(tensor, size):
    """
    Reshapes tensor so indices are defined like this:

    00, 01, 02, 03, 04, 05, 06, 07, 08, 09, size + 1, ..., self.max_size
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, size + 1, ..., self.max_size
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, size + 1, ..., self.max_size

    """
    return np.moveaxis(tensor, [0, 1], [1, 0])[:, 0: size].reshape(-1, *tensor.shape[2:])


class ReplayBuffer(S):
    """
    Storage class for Off-Policy algorithms.

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
    """

    # Data fields to store in buffer and contained in the generated batches
    storage_tensors = prl.DataTransitionKeys

    def __init__(self, size, device, actor, algorithm):

        self.actor = actor
        self.device = device
        self.algo = algorithm
        self.recurrent_actor = actor.is_recurrent
        self.max_size, self.size, self.step = size, 0, 0
        self.data = {k: None for k in self.storage_tensors}  # lazy init

        if self.recurrent_actor:
            self.sequence_length = algorithm.update_every
            self.overlap_length = int(actor.sequence_overlap * self.sequence_length)
            self.non_overlap_length = self.sequence_length - self.overlap_length
            self.max_size = (self.max_size // self.sequence_length) * self.sequence_length
            self.max_size *= 2  # to account for consecutive overlapping of sequences
            self.staged_overlap = None

        self.reset()

    @classmethod
    def create_factory(cls, size):
        """
        Returns a function that creates ReplayBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.

        Returns
        -------
        create_buffer_instance : func
            creates a new ReplayBuffer class instance.
        """

        def create_buffer(device, actor, algorithm):
            """Create and return a ReplayBuffer instance."""
            return cls(size, device, actor, algorithm)

        return create_buffer

    def init_tensors(self, sample):
        """
        Lazy initialization of data tensors from a sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """

        for k, v in sample.items():

            if k not in self.storage_tensors:
                continue

            if isinstance(v, dict):
                self.data[k] = {}
                for x, y in sample[k].items():
                    self.data[k][x] = np.zeros((self.max_size, *y.shape), dtype=np.float32)
            else:
                self.data[k] = np.zeros((self.max_size, *v.shape), dtype=np.float32)

    def get_data_slice(self, start_pos, end_pos):
        """
        Makes a copy of all tensors in the bufer between steps `start_pos`
        and `end_pos`.

        Parameters
        ----------
        start_pos : int
            initial slice position.
        end_pos : int
            final slice position.

        Returns
        -------
        data : dict
            data slice copied from the buffer.
        """

        copied_data = {k: None for k in self.storage_tensors}

        for k, v in self.data.items():

            if v is None:
                continue

            if isinstance(self.data[k], dict):
                copied_data[k] = {x: None for x in self.data[k]}
                for x, y in v.items():
                    copied_data[k][x] = np.copy(y[start_pos:end_pos])
            else:
                copied_data[k] = np.copy(v[start_pos:end_pos])

        return copied_data

    def get_all_buffer_data(self, data_to_cpu=False):
        """
        Return all currently stored data. If data_to_cpu, no need to do
        anything since data tensors are already in cpu memory.

        Parameters
        ----------
        data_to_cpu : bool
            Whether or not to move data tensors to cpu memory.

        Returns
        -------
        data : dict
            data currently stored in the buffer.
        """

        # Define data structure
        data = {k: None if not isinstance(self.data[k], dict) else
            {x: None for x in self.data[k]} for k in self.data}

        # If recurrent, get only whole sequences
        if self.recurrent_actor:
            idx = int((self.step // self.sequence_length) * self.sequence_length)
        else:
            idx = self.step

        # Fill up data
        for k, v in self.data.items():

            if v is None:
                continue

            if isinstance(self.data[k], dict):
                for x, y in self.data[k].items():
                    data[k][x] = y[:idx]
            else:
                data[k] = v[:idx]

        # If self.actor uses RNNs, save ending of last sequence
        if self.recurrent_actor and data_to_cpu:
            self.staged_overlap = self.get_data_slice(
                self.step - self.overlap_length, self.step)

        return data

    def reset(self):
        """
        Set class size and step to zero. If self.actor uses RNNs, add overlap
        slice of last sequence before reset at the beginning of the storage.
        """

        self.size -= self.step
        self.step = 0

        if self.recurrent_actor and self.staged_overlap:
            self.insert_data_slice(self.staged_overlap)

    def insert_data_slice(self, new_data):
        """
        Appends new_data to currently stored data.

        Parameters
        ----------
        new_data : dict
            Dictionary of env transition samples to be added to self.data.
        """

        lengths = []
        for k, v in new_data.items():

            if v is None:
                continue

            if isinstance(new_data[k], dict):

                if self.data[k] is None:
                    self.data[k] = {i: None for i in new_data[k].keys()}

                for x, y in new_data[k].items():
                    length = self.insert_single_tensor_slice(self.data[k], x, y)
                    lengths.append(length)
            else:
                length = self.insert_single_tensor_slice(self.data, k, v)
                lengths.append(length)

        assert len(set(lengths)) == 1

        self.step = (self.step + length) % self.max_size
        self.size = min(self.size + length, self.max_size)

    def insert_single_tensor_slice(self, tensor_storage, tensor_key, tensor_values):
        """
        Appends tensor_value to buffer dict using tensor_key as key.

        Parameters
        ----------
        tensor_storage :

        tensor_key : str
            key to use to store the tensor.
        tensor_values : np.ndarray
            tensor values.

        Returns
        -------
        l : int
            length (time axe) of the tensor added to the buffer.
        """

        l = tensor_values.shape[0]

        if tensor_storage[tensor_key] is None:  # If not defined, initialize tensor
            tensor_storage[tensor_key] = np.zeros((self.max_size, *tensor_values.shape[1:]), dtype=np.float32)

        if self.step + l <= self.max_size:  # If enough space, add tensor at the end
            tensor_storage[tensor_key][self.step:self.step + l] = tensor_values

        else:  # Circular buffer
            tensor_storage[tensor_key][
            self.step:self.max_size] = tensor_values[0:self.max_size - self.step]
            tensor_storage[tensor_key][0:l - self.max_size + self.step] = tensor_values[self.max_size - self.step:]

        return l

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
        if self.recurrent_actor and self.step % self.sequence_length == 0 and self.step != 0 and self.overlap_length > 0:
            next_seq_overlap = self.get_data_slice(self.step - self.overlap_length, self.step)
            self.insert_data_slice(next_seq_overlap)

        # Insert
        for k, v in sample.items():
            if isinstance(sample[k], dict):
                for x, y in sample[k].items():
                    self.data[k][x][self.step] = y.cpu()
            else:
                self.data[k][self.step] = v.cpu()

        # Update
        self.step = (self.step + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def before_gradients(self):
        """
        Steps required before updating actor policy model.
        """
        pass

    def after_gradients(self, batch, info):
        """
        Steps required after updating actor policy model

        Parameters
        ----------
        batch : dict
            Data batch used to compute the gradients.
        info : dict
            Additional relevant info from gradient computation.

        Returns
        -------
        info : dict
            info dict updated with relevant info from Storage.
        """
        return info

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

                # Fill up batch with sequences data
                for k, v in self.data.items():
                    # Only first recurrent state in each sequence needed
                    positions = seq_idxs * self.sequence_length if k in (prl.RHS, prl.RHS2) else idxs
                    if isinstance(v, dict):
                        for x, y in v.items():
                            t = dim0_reshape(y, self.size)[positions]
                            batch[k][x] = torch.as_tensor(t, dtype=torch.float32).to(self.device)
                    else:
                        t = dim0_reshape(v, self.size)[positions]
                        batch[k] = torch.as_tensor(t, dtype=torch.float32).to(self.device)
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
            setattr(self, parameter_name, new_parameter_value)
