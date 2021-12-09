import torch
import numpy as np
import pytorchrl as prl
from pytorchrl.agent.storages.off_policy.ere_buffer import EREBuffer as B


class HERBuffer(B):
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
    her_function : func
        Function to update obs, rhs, obs2 and rew according to HER paper.

    Warnings
    --------
    When using an environment vector of size larger than 1, episode sized must
    be of a fixed length. This HER implementation is not able to deal with envs
    of variable episode length, except in the case of environment vector size 1.

    """

    # Data fields to store in buffer and contained in the generated batches
    storage_tensors = prl.DataTransitionKeys

    def __init__(self, size, device, actor, algorithm, envs, her_function, n_step=1,
                 epsilon=0.0, alpha=0.0, beta=1.0, default_error=1000000, eta=1.0, cmin=5000):

        super(HERBuffer, self).__init__(
            size=size, device=device, actor=actor, algorithm=algorithm, envs=envs,
            n_step=n_step, epsilon=epsilon, alpha=alpha, beta=beta, default_error=default_error,
            eta=eta, cmin=cmin)

        self.her_function = her_function
        self.last_episode_start = 0

    @classmethod
    def create_factory(cls, size, her_function=lambda o, rhs, o2, rhs2, r, initial_state, final_state : (o, rhs, o2, rhs2, r),
                       n_step=1, epsilon=0.0, alpha=0.0, beta=1.0, default_error=1000000, eta=1.0, cmin=5000):
        """
        Returns a function that creates HERBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.

        Returns
        -------
        create_buffer_instance : func
            creates a new HERBuffer class instance.
        """

        def create_buffer(device, actor, algorithm, envs):
            """Create and return a HERBuffer instance."""
            return cls(size, device, actor, algorithm, her_function, n_step,
                       epsilon, alpha, beta, default_error, eta, cmin)

        return create_buffer

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

                if not self.recurrent_actor and k == prl.RHS2:
                    continue

                if isinstance(sample[k], dict):
                    for x, y in sample[k].items():
                        self.data[k][x][self.step] = y.cpu()
                else:
                    self.data[k][self.step] = sample[k].cpu()

            # Compute done and rew
            self.data[prl.REW][self.step], self.data[prl.DONE][
                self.step] = self._nstep_return()

            # Get obs, rhs and act from step buffer
            for k in (prl.OBS, prl.RHS, prl.ACT):

                if not self.recurrent_actor and k == prl.RHS:
                    continue

                tensor = self.n_step_buffer[k].popleft()
                if isinstance(tensor, dict):
                    for x, y in tensor.items():
                        self.data[k][x][self.step] = y.cpu()
                else:
                    self.data[k][self.step] = tensor.cpu()

            # Update
            self.step = (self.step + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

        # Handle end of episode - only works in the fixed episode length case!
        if (self.data[prl.DONE][self.step - 1] == 1.0).all():
            self.handle_end_of_episode()

    def copy_single_tensor(self, key, position):
        """Generates a copy of tensor `key` at index `position`."""

        if not self.recurrent_actor and key in (prl.RHS, prl.RHS2):
            position = 0

        if isinstance(self.data[key], dict):
            copied_data = {x: None for x in self.data[key]}
            for x, y in self.data[key].items():
                copied_data[x] = np.copy(y[position])
        else:
            copied_data = np.copy(self.data[key][position])

        return copied_data

    def handle_end_of_episode(self):
        """
        At the end of an environment episode, generates HER data and adds it
        to the replay buffer.
        """

        # Get sequence initial and final observations
        final_state = self.copy_single_tensor(prl.OBS, self.step - 1)
        initial_state = self.copy_single_tensor(prl.OBS, self.last_episode_start)

        # could also be self.step. I am choosing not to add the transition
        # in which obs and obs2 belong to different episodes
        current_step = self.step - 1

        # Re-create data according to HER
        for i in range(self.last_episode_start, current_step):

            obs, rhs, obs2, rhs2, rew = self.her_function(
                self.copy_single_tensor(prl.OBS, i),
                self.copy_single_tensor(prl.RHS, i),
                self.copy_single_tensor(prl.OBS2, i),
                self.copy_single_tensor(prl.RHS2, i),
                self.copy_single_tensor(prl.REW, i),
                initial_state,
                final_state)

            act = self.copy_single_tensor(prl.ACT, i)
            done = self.copy_single_tensor(prl.DONE, i)
            done2 = self.copy_single_tensor(prl.DONE2, i)

            sample = prl.DataTransition(obs, rhs, done, act, rew, obs2, rhs2, done2)._asdict()

            # Turn to tensors
            for k, v in sample.items():
                if isinstance(v, dict):
                    for x, y in k.items():
                        sample[k][x] = torch.as_tensor(y, dtype=torch.float32)
                else:
                    sample[k] = torch.as_tensor(k, dtype=torch.float32)

            # If using memory, save fixed length consecutive overlapping sequences
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

                    if not self.recurrent_actor and k == prl.RHS2:
                        continue

                    if isinstance(sample[k], dict):
                        for x, y in sample[k].items():
                            self.data[k][x][self.step] = y.cpu()
                    else:
                        self.data[k][self.step] = sample[k].cpu()

                # Compute done and rew
                self.data[prl.REW][self.step], self.data[prl.DONE][
                    self.step] = self._nstep_return()

                # Get obs, rhs and act from step buffer
                for k in (prl.OBS, prl.RHS, prl.ACT):

                    if not self.recurrent_actor and k == prl.RHS:
                        continue

                    tensor = self.n_step_buffer[k].popleft()
                    if isinstance(tensor, dict):
                        for x, y in tensor.items():
                            self.data[k][x][self.step] = y.cpu()
                    else:
                        self.data[k][self.step] = tensor.cpu()

                # Update
                self.step = (self.step + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)

        self.last_episode_start = self.step
