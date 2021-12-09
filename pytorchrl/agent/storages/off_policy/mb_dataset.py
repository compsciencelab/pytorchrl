import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorchrl as prl
from pytorchrl.agent.storages.base import Storage as S
import gym
from torch.nn.functional import one_hot


def dim0_reshape(tensor, size):
    """
    Reshapes tensor so indices are defined like this:

    00, 01, 02, 03, 04, 05, 06, 07, 08, 09, size + 1, ..., self.max_size
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, size + 1, ..., self.max_size
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, size + 1, ..., self.max_size

    """
    return np.moveaxis(tensor, [0, 1], [1, 0])[:, 0: size].reshape(-1, *tensor.shape[2:])


class MBReplayBuffer(S):
    """
    Storage class for Model Based algorithms.

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

    def __init__(self, size, validation_percentage, learn_reward_function, device, actor, algorithm, envs):

        self.actor = actor
        self.ensemble_size = actor.ensemble_size
        #self.scaler = actor.scaler
        self.validation_percentage = validation_percentage
        self.learn_reward_function = learn_reward_function
        self.ensemble_size = actor.ensemble_size
        self.device = device
        self.algo = algorithm
        self.max_size, self.size, self.step = size, 0, 0
        self.data = {k: None for k in self.storage_tensors}  # lazy init

        self.reset()

    @classmethod
    def create_factory(cls, size, validation_percentage, learn_reward_function):
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

        def create_buffer(device, actor, algorithm, envs):
            """Create and return a ReplayBuffer instance."""
            return cls(size, validation_percentage, learn_reward_function, device, actor, algorithm, envs)

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

        return data

    def reset(self):
        """
        Set class size and step to zero. If self.actor uses RNNs, add overlap
        slice of last sequence before reset at the beginning of the storage.
        """

        self.size -= self.step
        self.step = 0

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
        Steps required after updating actor policy model validation_percentage
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

    def generate_batches(self, num_mini_batch, mini_batch_size=256, num_epochs=1):
        """

        """

        # create dataloader with all obs, action, next_obs data
       
        for k, v in self.data.items():
            if k == prl.OBS:
                observations = v[:self.size]
            elif k == prl.ACT:
                actions = v[:self.size]
            elif k == prl.OBS2:
                next_observations = v[:self.size]
            elif k == prl.REW:
                rewards = v[:self.size]
            else:
                pass

        if type(self.actor.action_space) == gym.spaces.discrete.Discrete:
            actions = actions.astype(np.int64)
            actions = np.eye(actions.max()+1)[actions].squeeze(1)
            assert actions.shape == (observations.shape[0], 1, self.actor.action_space.n)

        inputs = np.concatenate((observations, actions), axis=-1)
        delta_state = next_observations - observations
        if self.learn_reward_function:
            labels = np.concatenate((delta_state, rewards), axis=-1)
        else:
            labels = delta_state
        num_validation = int(inputs.shape[0] * self.validation_percentage)

        train_inputs, train_labels = inputs[num_validation:], labels[num_validation:]
        holdout_inputs, holdout_labels = inputs[:num_validation], labels[:num_validation]

        # self.actor.scaler.fit(train_inputs)
        # train_inputs = self.actor.scaler.transform(train_inputs)
        # holdout_inputs = self.actor.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(self.device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(self.device)
        holdout_inputs = holdout_inputs.squeeze(1)
        holdout_labels = holdout_labels.squeeze(1)
        train_inputs = train_inputs.squeeze(1)
        train_labels = train_labels.squeeze(1)
        
        holdout_inputs = holdout_inputs[None, :, :].repeat(self.ensemble_size, 1, 1)
        holdout_labels = holdout_labels[None, :, :].repeat(self.ensemble_size, 1, 1)
        

        train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.ensemble_size)])
        max_batches = round((train_inputs.shape[0] / mini_batch_size), 0) - 1
        for batch_number, start_pos in enumerate(range(0, train_inputs.shape[0], mini_batch_size)):
            idx = train_idx[:, start_pos: start_pos + mini_batch_size]
            train_input = train_inputs[idx]
            train_label = train_labels[idx]
            train_input = torch.from_numpy(train_input).float().to(self.device)
            train_label = torch.from_numpy(train_label).float().to(self.device)
            batch = {"train_input": train_input,
                     "train_label": train_label,
                     "holdout_inputs": holdout_inputs,
                     "holdout_labels": holdout_labels,
                     "batch_number": batch_number,
                     "max_batches": max_batches}
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

