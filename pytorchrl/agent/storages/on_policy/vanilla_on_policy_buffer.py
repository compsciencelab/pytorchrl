import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler

import pytorchrl as prl
from pytorchrl.agent.storages.base import Storage as S


class VanillaOnPolicyBuffer(S):
    """
    Storage class for On-Policy algorithms.

    Parameters
    ----------
    size : int
        Storage capacity along time axis.
    device: torch.device
        CPU or specific GPU where data tensors will be placed and class
        computations will take place. Should be the same device where the
        actor model is located.
    actor : Actor
        Actor class instance.
    algorithm : Algorithm
        Algorithm class instance
    """

    # Data fields to store in buffer and contained in the generated batches
    storage_tensors = prl.OnPolicyDataKeys

    def __init__(self, size, device, actor, algorithm):

        self.actor = actor
        self.device = device
        self.algo = algorithm
        self.recurrent_actor = actor.is_recurrent
        self.max_size, self.size, self.step = size, 0, 0
        self.data = {k: None for k in self.storage_tensors}  # lazy init

        self.reset()

    @classmethod
    def create_factory(cls, size):
        """
        Returns a function that creates VanillaOnPolicyBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.

        Returns
        -------
        create_buffer_instance : func
            creates a new VanillaOnPolicyBuffer class instance.
        """
        def create_buffer_instance(device, actor, algorithm):
            """Create and return a VanillaOnPolicyBuffer instance."""
            return cls(size, device, actor, algorithm)
        return create_buffer_instance

    def init_tensors(self, sample):
        """
        Lazy initialization of data tensors from a sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """

        for k in sample:

            if k not in self.storage_tensors:
                continue

            # Handle dict sample value
            # init tensors for all dict entries
            if isinstance(sample[k], dict):
                self.data[k] = {}
                for x, v in sample[k].items():
                    self.data[k][x] = torch.zeros(self.max_size + 1, *v.shape).to(self.device)

            else: # Handle non dict sample value
                self.data[k] = torch.zeros(self.max_size + 1, *sample[k].shape).to(self.device)

        self.data[prl.RET] = self.data[prl.REW].clone()
        self.data[prl.ADV] = self.data[prl.VAL].clone()

    def get_all_buffer_data(self, data_to_cpu=False):
        """
        Return all currently stored data.

        Parameters
        ----------
        data_to_cpu : bool
            Whether or not to move data to cpu memory.
        """

        data = {k: v for k, v in self.data.items() if v is not None}

        if data_to_cpu:  # Move tensors to cpu
            for k in data:
                if isinstance(data[k], dict):
                    for x in data[k]:
                        data[k][x] = data[k][x].cpu()
                else:
                    data[k] = data[k].cpu()

        return data

    def insert_data_slice(self, new_data):
        """
        Replace currently stored data.

        Parameters
        ----------
        new_data : dict
            Dictionary of env transition samples to replace self.data with.
        """

        self.data = {k: v for k, v in new_data.items()}

        for k in new_data:

            # Move tensors to self.device
            if isinstance(self.data[k], dict):
                for x in self.data[k]:
                    self.data[k][x] = self.data[k][x].to(self.device)
            else:
                self.data[k] = self.data[k].to(self.device)

    def reset(self):
        """Set class counters to zero and remove stored data"""
        self.step, self.size = 0, 0

    def insert_transition(self, sample):
        """
        Store new transition sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """
        if self.size == 0 and self.data[prl.OBS] is None:  # data tensors lazy initialization
            self.init_tensors(sample)

        for k in sample:

            if k not in self.storage_tensors:
                continue

            # We use the same tensor to store obs and obs2
            # We also use single tensors dor rhs and rhs2,
            # and done and done2
            if k in (prl.OBS, prl.RHS, prl.DONE):
                pos = self.step + 1
                sample_k = "Next" + k
            else:
                pos = self.step
                sample_k = k

            # Copy sample tensor to buffer target position
            if isinstance(sample[k], dict):
                for x, v in sample[k].items():
                    self.data[k][x][pos].copy_(sample[sample_k][x])
            else:
                self.data[k][pos].copy_(sample[sample_k])

        self.step = (self.step + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def before_gradients(self):
        """
        Before updating actor policy model, compute returns and advantages.
        """

        last_tensors = {}
        for k in (prl.OBS, prl.RHS, prl.DONE):
            if isinstance(self.data[k], dict):
                last_tensors[k] = {x: self.data[k][x][self.step - 1] for x in self.data[k]}
            else:
                last_tensors[k] = self.data[k][self.step - 1]

        with torch.no_grad():
            _ = self.actor.get_action(last_tensors[prl.OBS], last_tensors[prl.RHS], last_tensors[prl.DONE])
            next_value, next_rhs = self.actor.get_value(last_tensors[prl.OBS], last_tensors[prl.RHS], last_tensors[prl.DONE])

        self.data[prl.RET][self.step].copy_(next_value)
        self.data[prl.VAL][self.step].copy_(next_value)

        if isinstance(next_rhs, dict):
            for x in self.data[prl.RHS]:
                self.data[prl.RHS][x][self.step].copy_(next_rhs[x])
        else:
            self.data[prl.RHS][self.step] = next_rhs

        self.compute_returns()
        self.compute_advantages()

    def after_gradients(self, batch, info):
        """
        After updating actor policy model, make sure self.step is at 0.

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

        for k in (prl.OBS, prl.RHS, prl.DONE):
            if isinstance(self.data[k], dict):
                for x in self.data[k]:
                    self.data[k][x][0].copy_(self.data[k][x][self.step - 1])
            else:
                self.data[k][0].copy_(self.data[k][self.step - 1])

        if self.step != 0:
            self.step = 0

        return info

    def compute_returns(self):
        """Compute return values."""
        gamma = self.algo.gamma
        len = self.step if self.step != 0 else self.max_size
        for step in reversed(range(len)):
            self.data[prl.RET][step] = (self.data[prl.RET][step + 1] * gamma * (
                1.0 - self.data[prl.DONE][step + 1]) + self.data[prl.REW][step])

    def compute_advantages(self):
        """Compute transition advantage values."""
        adv = self.data[prl.RET][:-1] - self.data[prl.VAL][:-1]
        self.data[prl.ADV] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def generate_batches(self, num_mini_batch, mini_batch_size, num_epochs=1, shuffle=True):
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
        shuffle : bool
            Whether to shuffle collected data or generate sequential

        Yields
        ------
        batch : dict
            Generated data batches.
        """

        num_proc = self.data[prl.DONE].shape[1]
        l = self.step if self.step != 0 else self.max_size

        if self.recurrent_actor:  # Batches for a feed recurrent actor

            num_envs_per_batch = num_proc // num_mini_batch
            assert num_proc >= num_mini_batch, "number processes greater than  mini batches"
            perm = torch.randperm(num_proc) if shuffle else list(range(num_proc))

            for _ in range(num_epochs):
                for start_ind in range(0, num_proc, num_envs_per_batch):

                    # Define batch structure
                    batch = {k: [] if not isinstance(self.data[k], dict) else
                    {x: [] for x in self.data[k]} for k in self.storage_tensors}

                    # Fill up batch with data
                    for offset in range(num_envs_per_batch):
                        ind = perm[start_ind + offset]
                        for k in batch:
                            # For recurrent state only first position needed
                            len = 1 if k == prl.RHS else l
                            if isinstance(self.data[k], dict):
                                for x in self.data[k]:
                                    batch[k][x].append(self.data[k][x][:len, ind])
                            else:
                                batch[k].append(self.data[k][:len, ind])

                    # Stack and reshape data to target shape
                    for k in batch:
                        if isinstance(self.data[k], dict):
                            for x in self.data[k]:
                                batch[k][x] = torch.stack(batch[k][x], dim=1).view(
                                    -1, *self.data[k][x].size()[2:])
                        else:
                            batch[k] = torch.stack(batch[k], dim=1).view(
                                -1, *self.data[k].size()[2:])

                    yield batch

        else:
            mini_batch_size = num_proc * l // num_mini_batch
            sampler = SubsetRandomSampler if shuffle else SequentialSampler
            for _ in range(num_epochs):
                for idxs in BatchSampler(sampler(range(num_proc * l)), mini_batch_size, drop_last=shuffle):

                    batch = {k: None for k in self.storage_tensors}

                    for k in batch:
                        if isinstance(self.data[k], dict):
                            tensor = {x: self.data[k][x][0:l].reshape(
                                -1, *self.data[k][x].shape[2:])[idxs] for x in self.data[k]}
                        else:
                            tensor = self.data[k][0:l].reshape(-1, *self.data[k].shape[2:])[idxs]
                        batch[k] = tensor

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
            setattr(self, parameter_name, new_parameter_value)
