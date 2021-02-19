import numpy as np
import torch

class HindsightExperienceReplayBuffer:
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
    """

    # Accepted data fields. Inserting other fields will raise AssertionError
    off_policy_data_fields = ("obs", "obs2", "rhs", "act", "rew", "done")

    def __init__(self, size, device, her_function=lambda o, rhs, o2, r, goal_state : (o, rhs, o2, r)):

        self.device = device
        self.max_size, self.size, self.step = size, 0, 0
        self.her_function = her_function
        self.data = {k: None for k in self.off_policy_data_fields}  # lazy init

        self.reset()

    @classmethod
    def create_factory(cls, size, her_function=lambda o, rhs, o2, r, goal_state : (o, rhs, o2, r)):
        """
        Returns a function that creates HindsightExperienceReplayBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.

        Returns
        -------
        create_buffer_instance : func
            creates a new OnPolicyBuffer class instance.
        """
        def create_buffer(device):
            """Create and return a ReplayBuffer instance."""
            return cls(size, device, her_function)
        return create_buffer

    def init_tensors(self, sample):
        """
        Lazy initialization of data tensors from a sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """
        assert set(sample.keys()) == set(self.data.keys())
        self.data["obs"] = np.zeros((self.max_size, *sample["obs"].shape), dtype=np.float32)
        self.data["obs2"] = np.zeros((self.max_size, *sample["obs"].shape), dtype=np.float32)
        self.data["rhs"] = np.zeros((self.max_size, *sample["rhs"].shape), dtype=np.float32)
        self.data["act"] = np.zeros((self.max_size, *sample["act"].shape), dtype=np.float32)
        self.data["rew"] = np.zeros((self.max_size, *sample["rew"].shape), dtype=np.float32)
        self.data["done"] = np.zeros((self.max_size, *sample["done"].shape), dtype=np.float32)

    def get_data(self, data_to_cpu=False):
        """Return currently stored data."""
        if data_to_cpu: data = {k: v for k, v in self.data.items() if v is not None}
        else: data = {k: v for k, v in self.data.items() if v is not None}
        return data

    def reset(self):
        """Set class counters to zero and remove stored data"""
        self.size -= self.step
        self.step, self.last_episode_start = 0, 0

    def add_data(self, new_data):
        """
        Appends new_data to currently stored data.

        Parameters
        ----------
        new_data : dict
            Dictionary of env transition samples to be added to self.data.
        """
        for k, v in new_data.items():

            if self.data[k] is None:
                self.data[k] = np.zeros((self.max_size, *v.shape[1:]), dtype=np.float32)

            len, num_proc = v.shape[0:2]

            if self.step + len <= self.max_size:
                self.data[k][self.step:self.step + len] = v
            else:
                self.data[k][self.step:self.max_size] = v[0:self.max_size - self.step]
                self.data[k][0:len - self.max_size + self.step] = v[self.max_size - self.step:]


        self.step = (self.step + len) % self.max_size
        self.size = min(self.size + len, self.max_size)

    def insert(self, sample):
        """
        Store new transition sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """
        if self.size == 0 and self.data["obs"] is None: # data tensors lazy initialization
            self.init_tensors(sample)

        # Add current observation
        self.data["obs"][self.step] = sample["obs"].cpu()
        self.data["rhs"][self.step] = sample["rhs"].cpu()
        self.data["act"][self.step] = sample["act"].cpu()
        self.data["rew"][self.step] = sample["rew"].cpu()
        self.data["obs2"][self.step] = sample["obs2"].cpu()
        self.data["done"][self.step] = sample["done"].cpu()

        self.step = (self.step + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        # Handle end of episode - fixed episode length case.
        if (sample["done"].cpu() == 1.0).all():

            goal_state = sample["obs2"].cpu()

            for i in range(self.last_episode_start, self.step):

                self.data["act"][self.step] = sample["act"].cpu()
                self.data["done"][self.step] = sample["done"].cpu()

                obs, rhs, obs2, rew = self.her_function(
                    sample["obs"].cpu(),
                    sample["rhs"].cpu(),
                    sample["obs2"].cpu(),
                    sample["rew"].cpu(),
                    goal_state)

                self.data["obs"][self.step] = obs
                self.data["rhs"][self.step] = rhs
                self.data["rew"][self.step] = rew
                self.data["obs2"][self.step] = obs2

                self.step = (self.step + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)

            self.last_episode_start = self.step

    def before_gradients(self, actor, algo):
        """
        Steps required before updating actor policy model.

        Parameters
        ----------
        actor : Actor class
            An actor class instance.
        algo : Algo class
            An algorithm class instance.
        """
        pass

    def after_gradients(self, actor, algo, batch, info):
        """
        Steps required after updating actor policy model

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
        pass

    def generate_batches(self, num_mini_batch, mini_batch_size, num_epochs=1,
                         recurrent_ac=False):
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
            Generated data batches.
        """
        num_proc = self.data["obs"].shape[1]

        if recurrent_ac:  # Batches for a feed recurrent actor
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
                yield {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in batch.items()}
