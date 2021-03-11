import numpy as np
import torch
from collections import deque
from pytorchrl.agent.storages.off_policy.per_buffer import PERBuffer as B


class EREBuffer(B):
    """
    Storage class for Off-Policy algorithms with Emphasizing Recent Experience
    buffer (https://arxiv.org/abs/1906.04009).

    This component extends PERBuffer, allowing to combine ERE with Prioritized
    Experience Replay (PER) if required. Nonetheless PER parameters, epsilon,
    alpha and beta, are set by default to values that make PER equivalent to
    a vanilla replay buffer, allowing to use only ERE. Also n step learning
    can be combined with PER and ERE using this component, but default n_step
    value is 1.

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

    def __init__(self, size, device, eta=1.0, cmin=5000, epsilon=0.0, alpha=0.0, beta=1.0, gamma=0.99, n_step=1):

        super(EREBuffer, self).__init__(
            size=size, device=device, gamma=gamma, n_step=n_step,
            epsilon=epsilon, alpha=alpha, beta=beta)

        self.eta = eta
        self.cmin = cmin
        self.initial_eta = eta
        self.eps_reward = deque(maxlen=20)
        self.max_grad_rew, self.min_grad_rew = - np.Inf, 0.0

    @classmethod
    def create_factory(cls, size, eta=1.0, cmin=5000, epsilon=0.0, alpha=0.0, beta=1.0, gamma=0.99, n_step=1):
        """
        Returns a function that creates EREBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.

        Returns
        -------
        create_buffer_instance : func
            creates a new EREBuffer class instance.
        """

        def create_buffer(device):
            """Create and return a EREBuffer instance."""
            return cls(size, device, eta, cmin, epsilon, alpha, beta, gamma, n_step)

        return create_buffer

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

        info = self.update_eta(info)
        return info

    def update_eta(self, info):
        """
        Adjust eta parameter based on how fast or slow the agent is learning
        in recent episodes
        """

        slopes = np.linspace(-1.0, 1.0, 1000)
        etas = np.linspace(1.0, self.initial_eta, 1000)
        if "performance/train_reward" in info.keys():
            self.eps_reward.append(info["performance/train_reward"])

        if len(self.eps_reward) == self.eps_reward.maxlen:
            reward_grad = np.gradient(self.eps_reward).mean()

            if reward_grad > self.max_grad_rew: self.max_grad_rew = reward_grad
            else: self.max_grad_rew *= 0.9999

            if reward_grad < - self.min_grad_rew: self.min_grad_rew = abs(reward_grad)
            else: self.min_grad_rew *= 0.9999

            if np.sign(reward_grad) == 1: reward_grad /= self.max_grad_rew
            else: reward_grad /= self.min_grad_rew

            idx = (np.abs(slopes - reward_grad)).argmin()
            self.eta = etas[idx]
            info.update({"algo/rew_grad": reward_grad})
            info.update({"algo/eta": self.eta})

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
        shuffle : bool
            Whether to shuffle collected data or generate sequential

        Yields
        ______
        batch : dict
            Generated data batches. Contains also extra information relevant to ERE.
        """
        num_proc = self.data["obs"].shape[1]
        N = num_proc * self.size

        if recurrent_ac:  # Batches to a feed recurrent actor
            raise NotImplementedError

        else:  # Batches for a feed forward actor
            for k in range(num_mini_batch):

                if num_proc * self.size < self.cmin:  # Standard
                    idxs = np.random.randint(0, num_proc * self.size, size=mini_batch_size)
                    weigths = 1.

                elif self.alpha == 0.0:  # ERE
                    ck = max(N * self.eta ** ((1000 * k) / num_mini_batch), self.cmin)
                    idxs = np.random.randint(N - ck, N, size=mini_batch_size)
                    weigths = 1.0

                else:  # PER + ERE
                    ck = max(N * self.eta ** ((1000 * k) / num_mini_batch), self.cmin)
                    priors = self.data["pri"][0:self.size].reshape(-1)[N - ck: N] + 1e-8
                    probs = priors / priors.sum()
                    idxs = np.random.choice(range(N - ck, N), size=mini_batch_size, p=probs)
                    weigths = np.power(ck * probs, - self.beta)
                    weigths /= weigths.max()

                batch =  dict(
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
