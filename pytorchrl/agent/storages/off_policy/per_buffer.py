import torch
import numpy as np
import pytorchrl as prl
from pytorchrl.agent.storages.off_policy.nstep_buffer import NStepReplayBuffer as B


def dim0_reshape(tensor, size):
    """
    Reshapes tensor so indices are defined like this:

    00, 01, 02, 03, 04, 05, 06, 07, 08, 09, size + 1, ..., self.max_size
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, size + 1, ..., self.max_size
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, size + 1, ..., self.max_size

    """
    return np.moveaxis(tensor, [0, 1], [1, 0])[:, 0: size].reshape(-1, *tensor.shape[2:])


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
    epsilon : float
        PER epsilon parameter.
    alpha : float
        PER alpha parameter.
    beta : float
        PER beta parameter.
    default_error : int or float
        Default TD error value to use for newly added data samples.
    """

    # Data fields to store in buffer and contained in the generated batches
    storage_tensors = prl.DataTransitionKeys

    def __init__(self, size, device, actor, algorithm, n_step=1, epsilon=0.0, alpha=0.0, beta=1.0, default_error=1000000):

        super(PERBuffer, self).__init__(
            size=size, device=device, actor=actor,
            algorithm=algorithm, n_step=n_step)

        self.beta = beta
        self.alpha = alpha
        self.epsilon = epsilon
        self.data["priority"] = None
        self.error = default_error

    @classmethod
    def create_factory(cls, size, n_step=1, epsilon=0.0, alpha=0.0, beta=1.0, default_error=1000000):
        """
        Returns a function that creates PERBuffer instances.

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

        Returns
        -------
        create_buffer_instance : func
            creates a new PERBuffer class instance.
        """

        def create_buffer(device, actor, algorithm):
            """Create and return a PERBuffer instance."""
            return cls(size, device, actor, algorithm, n_step, epsilon, alpha, beta, default_error)

        return create_buffer

    def get_priority(self, error):
        """Takes in the error of one or more examples and returns the proportional priority"""
        return np.power(error + self.epsilon, self.alpha)

    def get_sequence_priority(self, sequence_data, eta=0.9):
        """ _ """
        term1 = eta * np.max(sequence_data, axis=0)
        term2 = (1 - eta) * np.mean(sequence_data, axis=0)
        priority = term1 + term2
        return priority

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

        # if recurrent, save fixed-length consecutive overlapping sequences
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

    def before_gradients(self):
        """
        Steps required before updating actor policy model.
        """
        num_proc = self.data[prl.DONE].shape[1]

        if self.data["priority"] is None:  # Lazy initialization of priority
            if self.recurrent_actor:
                default_priority = self.get_sequence_priority(self.error * np.ones((self.sequence_length)))
            else:
                default_priority = self.error
            self.data["priority"] = default_priority * np.ones((self.max_size, num_proc, 1))

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

            assert "errors" in info[prl.ALGORITHM].keys(), "TD errors missing!"

            if self.recurrent_actor:

                endpos = int(self.size + self.sequence_length)

                # Get data indices and td errors
                idxs = np.array(batch.pop("idxs")).reshape(-1, self.sequence_length)
                errors = info[prl.ALGORITHM]["errors"].reshape(-1, self.non_overlap_length)

                # Since sequences overlap, update both current sequence and
                # start of the next overlapping sequence
                idxs = np.concatenate([idxs, idxs[:, :self.overlap_length] + self.overlap_length], axis=1)
                errors = torch.cat([errors, errors[:, :self.overlap_length]], dim=1)

                for i, e in zip(idxs, errors):  # each sequence in the batch

                    # Assign priorities to both end of current sequence
                    # and start of next sequence
                    dim0_reshape(self.data["priority"], endpos)[i] = \
                        self.get_priority(e.unsqueeze(1))

                    # Update current sequence average priority
                    sequence = dim0_reshape(self.data["priority"], endpos)[
                        i - self.overlap_length]
                    pri = self.get_sequence_priority(sequence)
                    dim0_reshape(self.data["priority"], endpos)[
                        i - self.overlap_length] = pri * np.ones(sequence.shape)

                    # Update next sequence average if some overlap
                    if self.overlap_length > 0:
                        sequence = dim0_reshape(
                            self.data["priority"], endpos)[i + self.non_overlap_length]
                        pri = self.get_sequence_priority(sequence)
                        dim0_reshape(self.data["priority"], endpos)[
                            i + self.non_overlap_length] = pri * np.ones(sequence.shape)
            else:
                self.data["priority"][0:self.size].reshape(-1, *self.data["priority"].shape[2:])[
                    batch.pop("idxs")] = self.get_priority(info[prl.ALGORITHM]["errors"])

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
                sequences_x_proc = int(self.size / self.sequence_length)

                assert self.size % self.sequence_length == 0, \
                    "Buffer does not contain an integer number of complete rollout sequences"

                # Define batch structure
                batch = {k: {} if not isinstance(self.data[k], dict) else
                    {x: {} for x in self.data[k]} for k in self.storage_tensors}

                # Select sequences
                if self.alpha == 0.0:
                    seq_idxs = np.random.randint(0, num_proc * sequences_x_proc, size=sequences_x_batch)
                    per_weigths = 1.0
                else:
                    priors = dim0_reshape(self.data["priority"], self.size)
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
                    probs = probs[self.sequence_length
                        - 1::self.sequence_length] * self.sequence_length
                    probs = np.split(probs, num_proc)
                    probs = [np.concatenate([chunk, np.zeros(
                        (1, 1))]) for chunk in probs]
                    ext_probs = np.concatenate(probs).squeeze(1)
                    seq_idxs = np.random.choice(range(len(ext_probs)), size=sequences_x_batch, p=ext_probs)

                # Get data indexes
                idxs = []
                for idx in seq_idxs:
                    idxs += range(idx * self.sequence_length, (idx + 1) * self.sequence_length)

                if not isinstance(per_weigths, float):
                    per_weigths = torch.as_tensor(per_weigths[idxs], dtype=torch.float32).to(self.device)

                # Fill up batch with data
                for k, v in self.data.items():
                    # Only first recurrent state in each sequence needed
                    positions = seq_idxs * self.sequence_length if k.startswith(prl.RHS) else idxs
                    if isinstance(v, dict):
                        for x, y in v.items():
                            t = dim0_reshape(y, self.size + self.sequence_length)[positions]
                            batch[k][x] = torch.as_tensor(t, dtype=torch.float32).to(self.device)
                    else:
                        t = dim0_reshape(v, self.size + self.sequence_length)[positions]
                        batch[k] = torch.as_tensor(t, dtype=torch.float32).to(self.device)

                batch.update({"per_weights": per_weigths, "n_step": self.n_step, "idxs": idxs})
                yield batch

            else:
                batch = {k: {} for k in self.storage_tensors}

                if self.alpha == 0.0:
                    idxs = np.random.randint(0, num_proc * self.size, size=mini_batch_size)
                    per_weigths = 1.0
                else:
                    priors = self.data["priority"][0:self.size].reshape(-1)
                    probs = priors / priors.sum()
                    idxs = np.random.choice(range(num_proc * self.size), size=mini_batch_size, p=probs)
                    per_weigths = np.power(num_proc * self.size * probs, -self.beta)
                    per_weigths = torch.as_tensor(per_weigths / per_weigths.max(), dtype=torch.float32).to(self.device)
                    per_weigths = per_weigths.view(-1, 1)[idxs]

                for k, v in self.data.items():
                    if isinstance(v, dict):
                        for x, y in v.items():
                            batch[k][x] = torch.as_tensor(y[0:self.size].reshape(
                                -1, *y.shape[2:])[idxs], dtype=torch.float32).to(self.device)
                    else:
                        batch[k] = torch.as_tensor(v[0:self.size].reshape(
                            -1, *v.shape[2:])[idxs], dtype=torch.float32).to(self.device)

                batch.update({"per_weights": per_weigths, "n_step": self.n_step, "idxs": idxs})
                yield batch
