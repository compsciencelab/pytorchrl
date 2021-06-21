import os
import ray
import time
import torch
import numpy as np
from collections import deque, defaultdict

import pytorchrl as prl
from pytorchrl.scheme.utils import check_message, pack
from pytorchrl.scheme.base.worker import Worker as W


class CWorker(W):
    """
     Worker class handling data collection.

    This class wraps an actor instance, a storage class instance and a
    train and a test vector environments. It collects data samples, sends
    them and and evaluates network versions.

    Parameters
    ----------
    index_worker : int
        Worker index.
    index_worker : int
        Index of gradient worker in charge of this data collection worker.
    algo_factory : func
        A function that creates an algorithm class.
    actor_factory : func
        A function that creates a policy.
    storage_factory : func
        A function that create a rollouts storage.
    fraction_samples : float
        Minimum fraction of samples required to stop if collection is
        synchronously coordinated and most workers have finished their
        collection task.
    compress_data_to_send : bool
        Whether or not to compress data before sending it to grad worker.
    train_envs_factory : func
        A function to create train environments.
    test_envs_factory : func
        A function to create test environments.
    initial_weights : ray object ID
        Initial model weights.
    device : str
        "cpu" or specific GPU "cuda:number`" to use for computation.

    Attributes
    ----------
    index_worker : int
        Index assigned to this worker.
    fraction_samples : float
        Minimum fraction of samples required to stop if collection is
        synchronously coordinated and most workers have finished their
        collection task.
    device : torch.device
        CPU or specific GPU to use for computation.
    compress_data_to_send : bool
        Whether or not to compress data before sending it to grad worker.
    actor : Actor
        An actor class instance.
    algo : Algo
        An algorithm class instance.
    envs_train : VecEnv
        A VecEnv class instance with the train environments.
    envs_test : VecEnv
        A VecEnv class instance with the test environments.
    storage : Storage
        A Storage class instance.
    iter : int
         Number of times samples have been collected and sent.
    actor_version : int
        Number of times the current actor version been has been updated.
    update_every : int
        Number of data samples to collect between network update stages.
    obs : torch.tensor
        Latest train environment observation.
    rhs : torch.tensor
        Latest policy recurrent hidden state.
    done : torch.tensor
        Latest train environment done flag.
    """

    def __init__(self,
                 index_worker,
                 index_parent,
                 algo_factory,
                 actor_factory,
                 storage_factory,
                 fraction_samples=1.0,
                 compress_data_to_send=False,
                 train_envs_factory=lambda x, y, z: None,
                 test_envs_factory=lambda v, x, y, z: None,
                 initial_weights=None,
                 device=None):

        super(CWorker, self).__init__(index_worker)
        self.index_worker = index_worker
        self.fraction_samples = fraction_samples

        # Computation device
        dev = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.compress_data_to_send = compress_data_to_send

        # Create Actor Critic instance
        self.actor = actor_factory(self.device)

        # Create Algorithm instance
        self.algo = algo_factory(self.device, self.actor)

        # Create Storage instance and set world initial state
        self.storage = storage_factory(self.device, self.actor, self.algo)

        # Define counters and other attributes
        self.iter, self.actor_version, self.samples_collected = 0, 0, 0
        self.update_every = self.algo.update_every or self.storage.max_size
        self.updates_per_iter = self.algo.num_mini_batch * self.algo.num_epochs

        # Create train environments
        self.envs_train = train_envs_factory(self.device, index_worker, index_parent)

        # Create test environments (if creation function available)
        self.envs_test = test_envs_factory(self.device, index_worker, index_parent, "test")

        if initial_weights:  # if remote worker

            # Set initial weights
            self.set_weights(initial_weights)

        if self.envs_train:
            # Define initial train states
            self.obs, self.rhs, self.done = self.actor.actor_initial_states(
                self.envs_train.reset())

            # Define reward tracking variable
            self.acc_reward = torch.zeros_like(self.done)

            # Collect initial samples
            print("Collecting initial samples...")
            self.collect_train_data(self.algo.start_steps)

        # Print worker information
        self.print_worker_info()

    def collect_data(self, listen_to=[], data_to_cpu=True):
        """
        Perform a data collection operation, returning rollouts and
        other relevant information about the process.

        Parameters
        ----------
        listen_to : list
            List of keywords to listen to trigger early stopping during
            collection.

        Returns
        -------
        data : dict
            Collected train data samples.
        info : dict
            Additional relevant information about the collection operation.
        """

        # Define dictionary to record relevant information
        info = {prl.EPISODES: {}, prl.TIME: {}, prl.VERSION: {}}

        # Collect train data
        t = time.time()
        train_info = self.collect_train_data(listen_to=listen_to)
        col_time = time.time() - t

        # Add information to info dict
        info[prl.EPISODES] = train_info
        info[prl.TIME][prl.COLLECTION] = col_time
        info[prl.VERSION][prl.COLLECTION] = self.actor_version
        info[prl.NUMSAMPLES] = self.samples_collected

        # Get new data collected
        data = self.storage.get_all_buffer_data(data_to_cpu)

        # Evaluate current network on test environments
        if self.envs_test and self.algo.num_test_episodes > 0:

            # Actor model updates performed so far
            updates_so_far = self.iter * self.updates_per_iter

            # Actor model updates after using current train data
            updates_next_time = (self.iter + 1) * self.updates_per_iter

            # If self.algo.test_every actor model updates will be reached using
            # current train data, evaluate actor
            updates = np.arange(updates_so_far, updates_next_time)
            if (updates % self.algo.test_every == 0).sum() > 0:
                    test_perf = self.evaluate()
                    info[prl.EPISODES]["TestReward"] = test_perf

        # Encode data if self.compress_data_to_send is True
        data_to_send = pack((data, info)) if self.compress_data_to_send else (data, info)

        # Reset storage and number of collected samples
        self.storage.reset()
        self.samples_collected = 0

        # Update counter
        self.iter += 1

        return data_to_send

    def collect_train_data(self, num_steps=None, listen_to=[]):
        """
        Collect train data from interactions with the environments.

        Parameters
        ----------
        num_steps : int
            Target number of train environment steps to take.
        listen_to : list

        Returns
        -------
        col_time : float
            Time, in seconds, spent in this operation.
        train_perf : float
            Average accumulated reward over recent train episodes.
        """

        info = defaultdict(list)

        # Define number of collections steps to perform
        num_steps = int(num_steps) if num_steps is not None else int(self.update_every)
        min_steps = int(num_steps * self.fraction_samples)

        for step in range(num_steps):

            # Predict next action, next rnn hidden state and
            # algo-specific outputs
            act, clip_act, rhs2, algo_data = self.algo.acting_step(
                self.obs, self.rhs, self.done)

            # Interact with env with predicted action (clipped
            # within action space)
            obs2, reward, done2, episode_infos = self.envs_train.step(clip_act)

            # Define transition sample
            transition = prl.DataTransition(
                self.obs, self.rhs, self.done, act, reward,
                obs2, rhs2, done2)._asdict()
            transition.update(algo_data)

            # Store transition in buffer
            self.storage.insert_transition(transition)

            # Handle end of episode - collect episode info
            done_positions = done2.nonzero()[:, 0].tolist()
            for i in done_positions:
                if "episode" in episode_infos[i]:  # gym envs should have it
                    for k, v in episode_infos[i]["episode"].items():
                        if isinstance(v, (float, int)):
                            if k == 'r':
                                k = "TrainReward"
                            info[k].append(v)

            # Update current world state
            self.obs, self.rhs, self.done = obs2, rhs2, done2

            # Keep track of num collected samples
            self.samples_collected += self.envs_train.num_envs

            # Check if stop message sent
            for l in listen_to:
                if check_message(l) == b"stop" and step >= min_steps:
                    break

        # Average episodes infos
        info = {} if len(info) == 0 else {k: np.mean(v) for k, v in info.items()}

        return info

    def evaluate(self):
        """
        Test current actor version in self.envs_test.

        Returns
        -------
        mean_test_perf : float
            Average accumulated reward over all tested episodes.
        """

        completed_episodes = []
        obs = self.envs_test.reset()
        rewards = np.zeros(obs.shape[0])
        obs, rhs, done = self.actor.actor_initial_states(obs)

        while len(completed_episodes) < self.algo.num_test_episodes:

            # Predict next action and rnn hidden state
            act, clip_act, rhs, _ = self.algo.acting_step(
                obs, rhs, done, deterministic=True)

            # Interact with env with predicted action (clipped
            # within action space)
            obs2, reward, done, _ = self.envs_test.step(clip_act)

            # Keep track of episode rewards and completed episodes
            rewards += reward.cpu().squeeze(-1).numpy()
            completed_episodes.extend(
                rewards[done.cpu().squeeze(-1).numpy() == 1.0].tolist())
            rewards[done.cpu().squeeze(-1).numpy() == 1.0] = 0.0

            obs = obs2

        return np.mean(completed_episodes)

    def set_weights(self, actor_weights):
        """
        Update the worker actor version with provided weights.

        actor_weights : dict of tensors
            Dict containing actor weights to be set.
        """
        self.actor_version = actor_weights[prl.VERSION]
        self.actor.load_state_dict(actor_weights[prl.WEIGHTS])

    def update_algorithm_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of self.algo, change its value to
        `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Algorithm attribute name
        new_parameter_value : float
            Algorithm new parameter value.
        """
        self.algo.update_algorithm_parameter(parameter_name, new_parameter_value)

    def update_storage_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of self.storage, change its value to
        `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Storage attribute name
        new_parameter_value : float
            Storage new parameter value.
        """
        self.storage.update_storage_parameter(parameter_name, new_parameter_value)

    def replace_agent_component(self, component_name, new_component_factory):
        """
        If `component_name` is an attribute of c_worker, replaces it with
        the component created by `new_component_factory`.

        Parameters
        ----------
        component_name : str
            Worker component name
        new_component_factory : func
            Function to create an instance of the new component.
        """

        # TODO. get component name from component itself

        if hasattr(self, component_name):
            if component_name == prl.ALGORITHM:
                new_component_component = new_component_factory(
                    self.device, self.actor)
            elif component_name == prl.ENV_TRAIN:
                new_component_component = new_component_factory(
                    self.device, self.index_worker)
            elif component_name == prl.ENV_TEST:
                new_component_component = new_component_factory(
                    self.device, self.index_worker, "test")
            else:
                new_component_component = new_component_factory(self.device)
            setattr(self, component_name, new_component_component)

