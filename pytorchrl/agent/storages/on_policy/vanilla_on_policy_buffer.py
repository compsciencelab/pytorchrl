import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
from ..base import Storage as S


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

    Attributes
    ----------
    max_size : int
        Storage capacity along time axis.
    device: torch.device
        CPU or specific GPU where data tensors will be placed and class
        computations will take place.
    """

    # Accepted data fields. Inserting other fields will raise AssertionError
    on_policy_data_fields = ("obs", "obs2", "rhs", "act", "rew", "val", "logp", "done")

    def __init__(self, size, device=torch.device("cpu")):

        self.device = device
        self.max_size, self.size, self.step = size, 0, 0
        self.data = {k: None for k in self.on_policy_data_fields}  # lazy init

        self.reset()

    @classmethod
    def create_factory(cls, size):
        """
        Returns a function that creates OnPolicyBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.

        Returns
        -------
        create_buffer_instance : func
            creates a new OnPolicyBuffer class instance.
        """
        def create_buffer_instance(device):
            """Create and return a OnPolicyBuffer instance."""
            return cls(size, device)
        return create_buffer_instance

    def init_tensors(self, sample):
        """
        Lazy initialization of data tensors from a sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """
        assert set(sample.keys()) == set(self.on_policy_data_fields)
        self.data["obs"] = torch.zeros(self.max_size + 1, *sample["obs"].shape).to(self.device)
        self.data["val"] = torch.zeros(self.max_size + 1, *sample["val"].shape).to(self.device)
        self.data["rhs"] = torch.zeros(self.max_size + 1, *sample["rhs"].shape).to(self.device)
        self.data["act"] = torch.zeros(self.max_size, *sample["act"].shape).to(self.device)
        self.data["rew"] = torch.zeros(self.max_size , *sample["rew"].shape).to(self.device)
        self.data["ret"] = torch.zeros(self.max_size + 1, *sample["rew"].shape).to(self.device)
        self.data["done"] = torch.zeros(self.max_size + 1, *sample["done"].shape).to(self.device)
        self.data["logp"] = torch.zeros(self.max_size, *sample["logp"].shape).to(self.device)

    def get_data(self, data_to_cpu=False):
        """Return currently stored data."""
        if data_to_cpu: data = {k: v for k, v in self.data.items() if v is not None}
        else: data = {k: v for k, v in self.data.items() if v is not None}
        return data

    def add_data(self, new_data):
        """
        Replace currently stored data.

        Parameters
        ----------
        new_data : dict
            Dictionary of env transition samples to replace self.data with.
        """
        for k, v in new_data.items(): self.data[k] = v

    def reset(self):
        """Set class counters to zero and remove stored data"""
        self.step, self.size = 0, 0

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

        self.data["obs"][self.step + 1].copy_(sample["obs2"])
        self.data["val"][self.step].copy_(sample["val"])
        self.data["rhs"][self.step + 1].copy_(sample["rhs"])
        self.data["act"][self.step].copy_(sample["act"])
        self.data["rew"][self.step].copy_(sample["rew"])
        self.data["done"][self.step + 1].copy_(sample["done"])
        self.data["logp"][self.step].copy_(sample["logp"])

        self.step = (self.step + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def before_gradients(self, actor, algo):
        """
        Before updating actor policy model, compute returns and advantages.

        Parameters
        ----------
        actor : Actor
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

        self.data["ret"][self.step] = next_value
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
        for step in reversed(range(len)):
            self.data["ret"][step] = (self.data["ret"][step + 1] * gamma * (
                1.0 - self.data["done"][step + 1]) + self.data["rew"][step])

    def compute_advantages(self):
        """Compute transition advantage values."""
        adv = self.data["ret"][:-1] - self.data["val"][:-1]
        self.data["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def generate_batches(self, num_mini_batch, mini_batch_size, num_epochs=1, recurrent_ac=False, shuffle=True):
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
        l = self.step if self.step != 0 else self.max_size

        if recurrent_ac:  # Batches for a feed recurrent actor
            num_envs_per_batch = num_proc // num_mini_batch
            assert num_proc >= num_mini_batch, "number processes greater than  mini batches"
            perm = torch.randperm(num_proc) if shuffle else list(range(num_proc))

            for _ in range(num_epochs):
                for start_ind in range(0, num_proc, num_envs_per_batch):
                    obs, rhs, act, val, ret, done, logp, adv = [], [], [], [], [], [], [], []

                    for offset in range(num_envs_per_batch):
                        ind = perm[start_ind + offset]
                        obs.append(self.data["obs"][:l, ind])
                        val.append(self.data["val"][:l, ind])
                        rhs.append(self.data["rhs"][0:1, ind])
                        act.append(self.data["act"][:l, ind])
                        ret.append(self.data["ret"][:l, ind])
                        done.append(self.data["done"][:l, ind])
                        logp.append(self.data["logp"][:l, ind])
                        if "adv" in self.data.keys():
                            adv.append(self.data["adv"][:l, ind])

                    batch = dict(
                        obs=torch.stack(obs, dim=1).view(-1, *self.data["obs"].size()[2:]),
                        val=torch.stack(val, dim=1).view(-1, *self.data["val"].size()[2:]),
                        rhs=torch.stack(rhs, dim=1).view(-1, *self.data["rhs"].size()[2:]),
                        act=torch.stack(act, dim=1).view(-1, *self.data["act"].size()[2:]),
                        ret=torch.stack(ret, dim=1).view(-1, *self.data["ret"].size()[2:]),
                        done=torch.stack(done, dim=1).view(-1, *self.data["done"].size()[2:]),
                        logp=torch.stack(logp, dim=1).view(-1, *self.data["logp"].size()[2:]),
                        adv=torch.stack(adv, dim=1).view(-1, *self.data["adv"].size()[2:])
                        if "adv" in self.data.keys() else None)

                    yield batch

        else:
            mini_batch_size = num_proc * l // num_mini_batch
            sampler = SubsetRandomSampler if shuffle else SequentialSampler
            for _ in range(num_epochs):
                for idxs in BatchSampler(sampler(range(num_proc * l)), mini_batch_size, drop_last=shuffle):
                    batch = dict(
                        obs=self.data["obs"][0:l].reshape(-1, *self.data["obs"].shape[2:])[idxs],
                        val=self.data["val"][0:l].reshape(-1, *self.data["val"].shape[2:])[idxs],
                        rhs=self.data["rhs"][0:l].reshape(-1, *self.data["rhs"].shape[2:])[idxs],
                        act=self.data["act"][0:l].reshape(-1, *self.data["act"].shape[2:])[idxs],
                        ret=self.data["ret"][0:l].reshape(-1, *self.data["ret"].shape[2:])[idxs],
                        done=self.data["done"][0:l].reshape(-1, *self.data["done"].shape[2:])[idxs],
                        logp=self.data["logp"][0:l].reshape(-1, *self.data["logp"].shape[2:])[idxs],
                        adv=self.data["adv"][0:l].reshape(-1, *self.data["adv"].shape[2:])[idxs]
                        if "adv" in self.data.keys() else None)

                    yield batch


