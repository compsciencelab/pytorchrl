import torch
import numpy as np
from pytorchrl.agent.storages.off_policy.nstep_buffer import NStepReplayBuffer as B


class PERBuffer(B):
    """
    Storage class for Off-Policy algorithms using PER (https://arxiv.org/abs/1707.01495).

    This component extends NStepReplayBuffer, enabling to combine PER with
    n step learning. However, default n_step value is 1, which is equivalent
    to not using n_step learning at all.

    Parameters
    ----------
    size : int
        Storage capacity along time axis.
    device: torch.device
        CPU or specific GPU where data tensors will be placed and class
        computations will take place. Should be the same device where the
        actor model is located.
    """

    # Accepted data fields. Inserting other fields will raise AssertionError
    off_policy_data_fields = ("obs", "obs2", "rhs", "act", "rew", "done")

    def __init__(self, size, device, epsilon=0.0, alpha=0.0, beta=1.0, gamma=0.99, n_step=1):

        super(PERBuffer, self).__init__(
            size=size, device=device, gamma=gamma, n_step=n_step)

        self.beta = beta
        self.alpha = alpha
        self.epsilon = epsilon
        self.data["pri"] = None

    @classmethod
    def create_factory(cls, size, epsilon=0.0, alpha=0.0, beta=1.0, gamma=0.99, n_step=1):
        """
        Returns a function that creates PERBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.

        Returns
        -------
        create_buffer_instance : func
            creates a new PERBuffer class instance.
        """

        def create_buffer(device):
            """Create and return a PERBuffer instance."""
            return cls(size, device, epsilon, alpha, beta, gamma, n_step)

        return create_buffer

    def init_tensors(self, sample, error=1000000):
        """
        Lazy initialization of data tensors from a sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        error : int or float
            Default TD error value to use for newly added data samples.
        """
        self.data["obs"] = np.zeros((self.max_size, *sample["obs"].shape), dtype=np.float32)
        self.data["obs2"] = np.zeros((self.max_size, *sample["obs"].shape), dtype=np.float32)
        self.data["rhs"] = np.zeros((self.max_size, *sample["rhs"].shape), dtype=np.float32)
        self.data["act"] = np.zeros((self.max_size, *sample["act"].shape), dtype=np.float32)
        self.data["rew"] = np.zeros((self.max_size, *sample["rew"].shape), dtype=np.float32)
        self.data["done"] = np.zeros((self.max_size, *sample["done"].shape), dtype=np.float32)
        self.data["pri"] = self.get_priority(error * np.ones((self.max_size, *sample["done"].shape), dtype=np.float32))
        self.num_proc = self.data["obs"].shape[1]

    def add_data(self, new_data, error=1000000):
        """
        Appends new_data to currently stored data.

        Parameters
        ----------
        new_data : dict
            Dictionary of env transition samples to be added to self.data.
        error : int or float
            Default TD error value to use for newly added data samples.
        """

        for k, v in new_data.items():
            len = v.shape[0]
            if self.data[k] is None:
                if k == "pri":
                    self.data[k] = self.get_priority(error * np.ones((self.max_size, *v.shape[1:]), dtype=np.float32))
                else:
                    self.data[k] = np.zeros((self.max_size, *v.shape[1:]), dtype=np.float32)
            if self.step + len <= self.max_size:
                self.data[k][self.step:self.step + len] = v
            else:
                self.data[k][self.step:self.max_size] = v[0:self.max_size - self.step]
                self.data[k][0:len - self.max_size + self.step] = v[self.max_size - self.step:]

        self.step = (self.step + len) % self.max_size
        self.size = min(self.size + len, self.max_size)

    def get_priority(self, error):
        """Takes in the error of one or more examples and returns the proportional priority"""
        return np.power(error + self.epsilon, self.alpha)

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

        Returns
        -------
        info : dict
            info dict updated with relevant info from Storage.
        """
        if "algo/errors" in info.keys():
            errors = info.pop("algo/errors")
            self.data["pri"][0:self.size].reshape(-1, *self.data["rew"].shape[2:])[
                batch["batch_idxs"]] = self.get_priority(errors)
        return info

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

        Yields
        ______
        batch : dict
            Generated data batches.
        """
        num_proc = self.data["obs"].shape[1]

        if recurrent_ac:  # Batches to a feed recurrent actor
            raise NotImplementedError

        else:  # Batches for a feed forward actor
            for _ in range(num_mini_batch):

                if self.alpha == 0.0:
                    idxs = np.random.randint(0, num_proc * self.size, size=mini_batch_size)
                    weigths = 1.0
                else:
                    priors = self.data["pri"][0:self.size].reshape(-1) + 1e-8
                    probs = priors / priors.sum()
                    idxs = np.random.choice(range(num_proc * self.size), size=mini_batch_size, p=probs)
                    weigths = np.power(num_proc * self.size * probs, -self.beta)
                    weigths /= weigths.max()
                    weigths = weigths.reshape(-1, 1)[idxs]

                batch = dict(
                    obs=self.data["obs"][0:self.size].reshape(-1, *self.data["obs"].shape[2:])[idxs],
                    rhs=self.data["rhs"][0:self.size].reshape(-1, *self.data["rhs"].shape[2:])[idxs],
                    act=self.data["act"][0:self.size].reshape(-1, *self.data["act"].shape[2:])[idxs],
                    rew=self.data["rew"][0:self.size].reshape(-1, *self.data["rew"].shape[2:])[idxs],
                    obs2=self.data["obs2"][0:self.size].reshape(-1, *self.data["obs2"].shape[2:])[idxs],
                    done=self.data["done"][0:self.size].reshape(-1, *self.data["done"].shape[2:])[idxs],
                    weigths=weigths)

                batch = {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in batch.items()}
                batch.update({"batch_idxs": idxs, "n_step": self.n_step})
                yield batch