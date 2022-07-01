from copy import deepcopy
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler

import pytorchrl as prl
from pytorchrl.utils import RunningMeanStd
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
        Algorithm class instance.
    envs : VecEnv
        Vector of environments instance.
    """

    # Data fields to store in buffer and contained in the generated batches
    storage_tensors = prl.OnPolicyDataKeys

    def __init__(self, size, device, actor, algorithm, envs):

        self.envs = envs
        if self.envs:
            self.num_envs = envs.num_envs
            self.frame_stack, self.frame_skip = 1, 0
            if "frame_stack" in self.envs.env_kwargs.keys():
                self.frame_stack = self.envs.env_kwargs["frame_stack"]
            if "frame_skip" in self.envs.env_kwargs.keys():
                self.frame_skip = self.envs.env_kwargs["frame_skip"]

        self.actor = actor
        self.device = device
        self.algo = algorithm
        self.recurrent_actor = actor.is_recurrent
        self.max_size, self.size, self.step = int(size), 0, 0
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

        def create_buffer_instance(device, actor, algorithm, envs):
            """Create and return a VanillaOnPolicyBuffer instance."""
            return cls(size, device, actor, algorithm, envs)

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

            if not self.recurrent_actor and k in (prl.RHS, prl.RHS2):
                size = 0
            else:
                size = self.max_size

            # Handle dict sample value
            # init tensors for all dict entries
            if isinstance(sample[k], dict):
                self.data[k] = {}
                for x, v in sample[k].items():
                    self.data[k][x] = torch.zeros(size + 1, *v.shape).to(self.device)

            else:  # Handle non dict sample value
                self.data[k] = torch.zeros(size + 1, *sample[k].shape).to(self.device)

        self.data[prl.RET] = deepcopy(self.data[prl.REW])
        self.data[prl.ADV] = deepcopy(self.data[prl.VAL])

        # In case of an algorithm with intrinsic rewards
        if prl.IREW in sample.keys() and prl.IVAL in sample.keys():
            self.data[prl.IRET] = deepcopy(self.data[prl.IREW])
            self.data[prl.IADV] = deepcopy(self.data[prl.IVAL])
            self.int_reward_rms = RunningMeanStd(shape=(1,), device=self.device)
            self.data[prl.MASK] = torch.ones_like(self.data[prl.DONE])
            self.storage_tensors += (prl.MASK,)

    def get_num_channels_obs(self, sample):
        """
        Obtain num_channels_obs and set it as class attribute.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """
        self.num_channels_obs = int(sample[prl.OBS][0].shape[0] // self.frame_stack)

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
            self.get_num_channels_obs(sample)

        for k in sample:

            if k not in self.storage_tensors:
                continue

            if not self.recurrent_actor and k in (prl.RHS, prl.RHS2):
                continue

            # We use the same tensor to store obs and obs2
            # We also use single tensors for rhs and rhs2,
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

        # Get most recent state
        last_tensors = {}
        step = self.step if self.step != 0 else -1
        for k in (prl.OBS, prl.RHS, prl.DONE):
            if isinstance(self.data[k], dict):
                last_tensors[k] = {x: self.data[k][x][step] for x in self.data[k]}
            else:
                last_tensors[k] = self.data[k][step]

        # Predict values given most recent state
        with torch.no_grad():
            _ = self.actor.get_action(last_tensors[prl.OBS], last_tensors[prl.RHS], last_tensors[prl.DONE])
            value_dict = self.actor.get_value(last_tensors[prl.OBS], last_tensors[prl.RHS], last_tensors[prl.DONE])
            next_rhs = value_dict.get("rhs")

        # Store next recurrent hidden state
        if isinstance(next_rhs, dict):
            for x in self.data[prl.RHS]:
                self.data[prl.RHS][x][step].copy_(next_rhs[x])
        else:
            self.data[prl.RHS][step] = next_rhs

        # Compute returns and advantages
        if isinstance(self.data[prl.VAL], dict):
            # If multiple critics
            for x in self.data[prl.VAL]:
                self.data[prl.VAL][x][step].copy_(value_dict.get(x))
                self.compute_returns(
                    self.data[prl.REW], self.data[prl.RET][x], self.data[prl.VAL][x], self.data[prl.DONE],
                    self.algo.gamma)
                self.data[prl.ADV][x] = self.compute_advantages(self.data[prl.RET][x], self.data[prl.VAL][x])
        else:
            # If single critic
            self.data[prl.VAL][step].copy_(value_dict.get("value_net1"))
            self.compute_returns(
                self.data[prl.REW], self.data[prl.RET], self.data[prl.VAL], self.data[prl.DONE], self.algo.gamma)
            self.data[prl.ADV] = self.compute_advantages(self.data[prl.RET], self.data[prl.VAL])

        if hasattr(self.algo, "gamma_intrinsic") and prl.IREW in self.data.keys():
            self.normalize_int_rewards()
            self.algo.state_rms.update(
                self.data[prl.OBS][:, :, -self.num_channels_obs:, ...].reshape(-1, 1, *self.data[prl.OBS].shape[3:]))

        # If algorithm with intrinsic rewards, also compute ireturns and iadvantages
        if prl.IVAL in self.data.keys() and prl.IREW in self.data.keys():
            if isinstance(self.data[prl.IVAL], dict):
                # If multiple critics
                for x in self.data[prl.IVAL]:
                    self.data[prl.IVAL][x][step].copy_(value_dict.get(x))
                    self.compute_returns(
                        self.data[prl.IREW], self.data[prl.IRET][x], self.data[prl.IVAL][x],
                        torch.zeros_like(self.data[prl.DONE]), self.algo.gamma_intrinsic)
                    self.data[prl.IADV][x] = self.compute_advantages(self.data[prl.IRET][x], self.data[prl.IVAL][x])
            else:
                # If single critic
                self.data[prl.IVAL][step].copy_(value_dict.get("ivalue_net1"))
                self.compute_returns(
                    self.data[prl.IREW], self.data[prl.IRET], self.data[prl.IVAL],
                    torch.zeros_like(self.data[prl.DONE]), self.algo.gamma_intrinsic)
                self.data[prl.IADV] = self.compute_advantages(self.data[prl.IRET], self.data[prl.IVAL])

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

        step = self.step if self.step != 0 else -1
        for k in (prl.OBS, prl.RHS, prl.DONE):
            if isinstance(self.data[k], dict):
                for x in self.data[k]:
                    self.data[k][x][0].copy_(self.data[k][x][step])
            else:
                self.data[k][0].copy_(self.data[k][step])

        if self.step != 0:
            self.step = 0

        return info

    def compute_returns(self, rewards, returns, values, dones, gamma):
        """Compute return values."""
        length = self.step - 1 if self.step != 0 else self.max_size
        returns[length].copy_(values[length])
        for step in reversed(range(length)):
            returns[step] = (returns[step + 1] * gamma * (1.0 - dones[step + 1]) + rewards[step])

    def compute_advantages(self, returns, values):
        """Compute transition advantage values."""
        adv = returns[:-1] - values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        return adv

    def normalize_int_rewards(self):
        """
        In order to keep the rewards on a consistent scale, the intrinsic rewards are normalized by dividing
        them by a running estimate of their standard deviation.
        """
        gamma = self.algo.gamma_intrinsic
        length = self.step - 1 if self.step != 0 else self.max_size
        rewems = torch.zeros_like(self.data[prl.RET])
        for step in reversed(range(length)):
            rewems[step] = rewems[step + 1] * gamma + self.data[prl.IREW][step]
        self.int_reward_rms.update(rewems[0:-1].reshape(-1, 1))
        self.data[prl.IREW] = self.data[prl.IREW] / (self.int_reward_rms.var.float() ** 0.5)
        # self.data[prl.IREW] = self.data[prl.IREW] / (self.int_reward_rms.var ** 0.5)

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
        l = self.step - 1 if self.step != 0 else self.max_size

        if self.recurrent_actor:  # Batches for a feed recurrent actor

            num_envs_per_batch = num_proc // num_mini_batch
            assert num_proc >= num_mini_batch, "number processes greater than  mini batches"

            for _ in range(num_epochs):
                perm = torch.randperm(num_proc) if shuffle else list(range(num_proc))
                for start_ind in range(0, num_proc, num_envs_per_batch):

                    # Define batch structure
                    batch = {k: [] if not isinstance(self.data[k], dict) else
                    {x: [] for x in self.data[k]} for k in self.data.keys()}

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
                for samples in BatchSampler(sampler(range(num_proc * l)), mini_batch_size, drop_last=shuffle):

                    batch = {k: None for k in self.data.keys()}

                    for k in batch:

                        if k in (prl.RHS, prl.RHS2):
                            size, idxs = 1, torch.tensor([0])
                        else:
                            size, idxs = l, samples

                        if isinstance(self.data[k], dict):
                            tensor = {x: self.data[k][x][0:size].reshape(
                                -1, *self.data[k][x].shape[2:])[idxs] for x in self.data[k]}
                        else:
                            tensor = self.data[k][0:size].reshape(-1, *self.data[k].shape[2:])[idxs]

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
