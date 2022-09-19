import numpy as np
import torch
from collections import deque
import pytorchrl as prl
from pytorchrl.agent.storages.off_policy.per_buffer import PERBuffer as B


def dim0_reshape(tensor, size1, size2):
    """
    Reshapes tensor so indices are defined like this:

    00, 01, 02, 03, 04, 05, 06, 07, 08, 09, size + 1, ..., self.max_size
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, size + 1, ..., self.max_size
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, size + 1, ..., self.max_size

    """
    return np.moveaxis(tensor, [0, 1], [1, 0])[:, size1: size2].reshape(-1, *tensor.shape[2:])


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
    envs : VecEnv
        Vector of environments instance.
    actor : Actor
        Actor class instance.
    algorithm : Algorithm
        Algorithm class instance.
    n_step : int or float
        Number of future steps used to computed the truncated n-step return value.
    epsilon : float
        PER epsilon parameter.
    alpha : float
        PER alpha parameter.
    beta : float
        PER beta parameter.
    default_error : int or float
        Default TD error value to use for newly added data samples.
    eta : float
        ERE eta parameter.
    cmin : int
        ERE cmin parameter.
    """

    # Data fields to store in buffer and contained in the generated batches
    storage_tensors = prl.OffPolicyDataKeys

    def __init__(self, size, device, actor, algorithm, envs, n_step=1, epsilon=0.0, alpha=0.0, beta=1.0, default_error=1000000, eta=1.0, cmin=5000):

        super(EREBuffer, self).__init__(
            size=size, device=device, actor=actor, algorithm=algorithm,
            envs=envs, n_step=n_step, epsilon=epsilon, alpha=alpha, beta=beta,
            default_error=default_error)

        self.eta = eta
        self.cmin = cmin
        self.initial_eta = eta
        self.eps_reward = deque(maxlen=20)
        self.max_grad_rew, self.min_grad_rew = - np.Inf, 0.0

    @classmethod
    def create_factory(cls, size, n_step=1, epsilon=0.0, alpha=0.0, beta=1.0, default_error=1000000, eta=1.0, cmin=5000):
        """
        Returns a function that creates EREBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.
        n_step : int or float
            Number of future steps used to computed the truncated n-step return value.
        epsilon : float
            PER epsilon parameter.
        alpha : float
            PER alpha parameter.
        beta : float
            PER beta parameter.
        default_error : int or float
            Default TD error value to use for newly added data samples.
        eta : float
            ERE eta parameter.
        cmin : int
            ERE cmin parameter.

        Returns
        -------
        create_buffer_instance : func
            creates a new EREBuffer class instance.
        """

        def create_buffer(device, actor, algorithm, envs):
            """Create and return a EREBuffer instance."""
            return cls(size, device, actor, algorithm, envs, n_step, epsilon, alpha, beta, default_error, eta, cmin)

        return create_buffer

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

        if "per_weights" in batch.keys() and isinstance(batch["per_weights"], torch.Tensor):

            num_proc = self.data[prl.DONE].shape[1]

            if self.recurrent_actor:

                N = self.size // self.sequence_length
                ck = batch.pop("ck")
                start = int((N - ck) * self.sequence_length)
                end = int(self.size + self.sequence_length)

                # Get data indices and td errors
                idxs = np.array(batch.pop("idxs")).reshape(-1, self.sequence_length)
                errors = info[prl.ALGORITHM]["errors"].reshape(-1, self.non_overlap_length)

                # Since sequences overlap, update both current sequence and
                # start of the next overlapping sequence
                idxs = np.concatenate([idxs, idxs[:, :self.overlap_length] + self.overlap_length], axis=1)
                errors = torch.cat([errors, errors[:, :self.overlap_length]], dim=1)

                for i, e in zip(idxs, errors):

                    # Assign priorities to both end of current sequence
                    # and start of next sequence
                    dim0_reshape(self.data["priority"], start, end
                                 )[i] = self.get_priority(e.unsqueeze(1))

                    # Update current sequence average priority
                    sequence = dim0_reshape(self.data["priority"], start, end)[
                        i - self.overlap_length]
                    pri = self.get_sequence_priority(sequence)
                    dim0_reshape(self.data["priority"], start, end)[i -
                        self.overlap_length] = pri * np.ones(sequence.shape)

                    # Update next sequence average if some overlap
                    if self.overlap_length > 0:
                        sequence = dim0_reshape(self.data["priority"],
                            start, end)[i + self.non_overlap_length]
                        pri = self.get_sequence_priority(sequence)
                        dim0_reshape(self.data["priority"], start, end)[i +
                            self.non_overlap_length] = pri * np.ones(sequence.shape)

            else:
                N = self.size * num_proc
                ck = batch.pop("ck")
                self.data["priority"][0:self.size].reshape(-1, *self.data[
                    "priority"].shape[2:])[N - ck: N][batch.pop("idxs")] = \
                        self.get_priority(info[prl.ALGORITHM]["errors"])

        info = self.update_eta(info)
        return info

    def update_eta(self, info):
        """
        Adjust eta parameter based on how fast or slow the agent is learning
        in recent episodes
        """

        slopes = np.linspace(-1.0, 1.0, 1000)
        etas = np.linspace(1.0, self.initial_eta, 1000)
        if prl.EPISODES in info.keys() and "TrainReward" in info[prl.EPISODES].keys():
            self.eps_reward.append(info[prl.EPISODES]["TrainReward"])

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
            info[prl.ALGORITHM].update({"RewardGradient": reward_grad})
            info[prl.ALGORITHM].update({"eta": self.eta})

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
            Generated data batches. Contains also extra information relevant to ERE.
        """
        num_proc = self.data[prl.DONE].shape[1]

        if self.recurrent_actor:  # Batches to a feed recurrent actor

            for k in range(num_mini_batch):

                # Define batch structure
                batch = {k: {} if not isinstance(self.data[k], dict) else
                {x: {} for x in self.data[k]} for k in self.data.keys()}

                sequences_x_batch = mini_batch_size // self.sequence_length + 1
                sequences_x_proc = int(self.size / self.sequence_length)
                N = sequences_x_proc

                assert self.size % self.sequence_length == 0, \
                    "Buffer does not contain an integer number of complete rollout sequences"

                if num_proc * self.size < self.cmin:  # Standard

                    seq_idxs = np.random.randint(0, num_proc * sequences_x_proc, size=sequences_x_batch)
                    per_weigths, ck = 1.0, N
                    start, end = int((N - ck) * self.sequence_length), int(self.size)

                elif self.alpha == 0.0:  # ERE

                    cmin = self.cmin // self.sequence_length // num_proc
                    ck = int(max(N * self.eta ** ((1000 * k) / num_mini_batch), cmin))

                    # reshape between N - ck, N
                    start, end = int((N - ck) * self.sequence_length), int(self.size)

                    seq_idxs = np.random.randint(0, ck * num_proc, size=sequences_x_batch)
                    per_weigths = 1.0

                else:  # PER + ERE

                    cmin = self.cmin // self.sequence_length // num_proc
                    ck = max(N * self.eta ** ((1000 * k) / num_mini_batch), cmin)

                    # reshape between N - ck, N
                    start = int((N - ck) * self.sequence_length)
                    end = int(self.size)

                    priors = dim0_reshape(self.data["priority"], start, end)
                    probs = priors / priors.sum()

                    per_weigths = np.power(num_proc * self.size * probs, - self.beta)
                    per_weigths = per_weigths / per_weigths.max()

                    # Trick to allow updating priorities of next overlapping
                    # sequences after gradient computation. Insert some "0.0"
                    # values in per_weights for end-of-row + 1 sequences.
                    per_weigths = np.split(per_weigths, num_proc)
                    per_weigths = [np.concatenate([chunk, np.zeros((
                        self.sequence_length, 1))]) for chunk in per_weigths]
                    per_weigths = np.concatenate(per_weigths)

                    # Trick to allow updating priorities of next overlapping
                    # sequences after gradient computation. Insert some "0.0"
                    # values in probs for end-of-row + 1 sequences.
                    probs = probs[int((N - ck + 1) * self.sequence_length) - 1
                        ::self.sequence_length] * self.sequence_length
                    probs = np.split(probs, num_proc)
                    probs = [np.concatenate([chunk, np.zeros((1, 1))]) for chunk in probs]
                    ext_probs = np.concatenate(probs).squeeze(1)

                    end += self.sequence_length
                    seq_idxs = np.random.choice(range(
                        len(ext_probs)), size=sequences_x_batch, p=ext_probs)

                # Get data indexes
                idxs = []
                for idx in seq_idxs:
                    idxs += range(idx * self.sequence_length, (idx + 1) * self.sequence_length)

                if not isinstance(per_weigths, float):
                    per_weigths = torch.as_tensor(per_weigths[idxs], dtype=torch.float32).to(self.device)

                # Fill up batch with data
                for k, v in self.data.items():
                    # Only first recurrent state in each sequence needed
                    positions = seq_idxs * self.sequence_length if k in (prl.RHS, prl.RHS2) else idxs
                    if isinstance(v, dict):
                        for x, y in v.items():
                            t = dim0_reshape(y, start, end)[positions]
                            batch[k][x] = torch.as_tensor(t, dtype=torch.float32).to(self.device)
                    else:
                        t = dim0_reshape(v, start, end)[positions]
                        batch[k] = torch.as_tensor(t, dtype=torch.float32).to(self.device)

                batch.update({"per_weights": per_weigths, "n_step": self.n_step, "idxs": idxs, "ck": ck})
                yield batch

        else:  # Batches for a feed forward actor
            for k in range(num_mini_batch):

                batch = {k: {} for k in self.data.keys()}
                N = num_proc * self.size
                per_weigths = None

                if num_proc * self.size < self.cmin:  # Standard
                    samples = np.random.randint(0, num_proc * self.size, size=mini_batch_size)
                    ck = N

                elif self.alpha == 0.0:  # ERE
                    ck = int(max(N * self.eta ** ((1000 * k) / num_mini_batch), self.cmin))
                    samples = np.random.randint(ck, size=mini_batch_size)

                else:  # PER + ERE
                    ck = int(max(N * self.eta ** ((1000 * k) / num_mini_batch), self.cmin))
                    priors = self.data["priority"][0:self.size].reshape(-1, 1)[N - ck: N]
                    probs = priors / priors.sum()
                    samples = np.random.choice(range(ck), size=mini_batch_size, p=probs.squeeze(1))

                    per_weigths = np.power(ck * probs, - self.beta)
                    per_weigths /= per_weigths.max()
                    per_weigths = per_weigths[samples]
                    per_weigths = torch.as_tensor(per_weigths, dtype=torch.float32).to(self.device)

                for k, v in self.data.items():

                    if k in (prl.RHS, prl.RHS2):
                        size, idxs = 1, np.array([0])
                    else:
                        size, idxs = self.size, samples

                    if isinstance(v, dict):
                        for x, y in v.items():
                            batch[k][x] = torch.as_tensor(y[0:self.size].reshape(
                                -1, *y.shape[2:])[N - ck: N][idxs], dtype=torch.float32).to(self.device)
                    else:
                        batch[k] = torch.as_tensor(v[0:self.size].reshape(
                            -1, *v.shape[2:])[N - ck: N][idxs], dtype=torch.float32).to(self.device)

                batch.update({"n_step": self.n_step, "idxs": idxs, "ck": ck})

                if per_weigths is not None:
                    batch.update({"per_weights": per_weigths})

                yield batch

