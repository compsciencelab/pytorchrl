import ray
import time
import torch
import numpy as np
from collections import deque
from ..base.worker import Worker as W
from ..utils import check_message


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

        # Create Actor Critic instance
        self.actor = actor_factory(self.device)

        # Create Algorithm instance
        self.algo = algo_factory(self.device, self.actor)

        # Create Storage instance and set world initial state
        self.storage = storage_factory(self.device)

        # Define counters and other attributes
        self.iter, self.actor_version, self.samples_collected = 0, 0, 0
        self.update_every = self.algo.update_every or self.storage.max_size

        # Create train environments
        self.envs_train = train_envs_factory(self.device, index_worker, index_parent)

        # Create test environments (if creation function available)
        self.envs_test = test_envs_factory(self.device, index_worker, index_parent, "test")

        if initial_weights:  # if remote worker

            # Set initial weights
            self.set_weights(initial_weights)

        if self.envs_train:
            # Define initial train states
            self.obs, self.rhs, self.done = self.actor.policy_initial_states(
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

        # Collect train data
        col_time, train_perf = self.collect_train_data(listen_to=listen_to)

        # Get collected rollouts
        data = self.storage.get_data(data_to_cpu)
        self.storage.reset()

        # Add information to info dict
        info = {}
        info.update({"debug/collect_time": col_time})
        info.update({"col_version": self.actor_version})
        info.update({"collected_samples": self.samples_collected})
        if train_perf: info.update({"performance/train_reward": train_perf})
        self.samples_collected = 0

        # Evaluate current network on test environments
        if self.iter % self.algo.test_every == 0:
            if self.envs_test and self.algo.num_test_episodes > 0:
                test_perf = self.evaluate()
                info.update({"performance/test_reward": test_perf})

        # Update counter
        self.iter += 1

        return data, info

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
        t = time.time()
        train_perf = []
        num_steps = num_steps if num_steps is not None else int(self.update_every)
        min_steps = int(num_steps * self.fraction_samples)

        for step in range(num_steps):

            # Predict next action, next rnn hidden state and algo-specific outputs
            act, clip_act, rhs, algo_data = self.algo.acting_step(
                self.obs, self.rhs, self.done)

            # Interact with envs_vector with predicted action (clipped within action space)
            obs2, reward, done, infos = self.envs_train.step(clip_act)

            # Handle end of episode
            self.acc_reward += reward
            ended_eps = self.acc_reward[done == 1.0].tolist()
            if len(ended_eps) > 0: train_perf.append(np.mean(ended_eps))
            self.acc_reward[done == 1.0] = 0.0

            # Prepare transition dict
            transition = {"obs": self.obs, "rhs": rhs, "act": act, "rew": reward, "obs2": obs2, "done": done}
            transition.update(algo_data)

            # Store transition in buffer
            self.storage.insert(transition)

            # Update current world state
            self.obs, self.rhs, self.done = obs2, rhs, done

            # Keep track of num collected samples
            self.samples_collected += self.envs_train.num_envs

            # Check if stop message sent
            for l in listen_to:
                if check_message(l) == b"stop" and step >= min_steps:
                    break

        col_time = time.time() - t
        train_perf = None if len(train_perf) == 0 else np.mean(train_perf)

        return col_time, train_perf

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
        obs, rhs, done = self.actor.policy_initial_states(obs)

        while len(completed_episodes) < self.algo.num_test_episodes:
            # Predict next action and rnn hidden state
            act, clip_act, rhs, _ = self.algo.acting_step(
                obs, rhs, done, deterministic=True)

            # Interact with env with predicted action (clipped within action space)
            obs2, reward, done, _ = self.envs_test.step(clip_act)

            # Keep track of episode rewards and completed episodes
            rewards += reward.cpu().squeeze(-1).numpy()
            completed_episodes.extend(
                rewards[done.cpu().squeeze(-1).numpy() == 1.0].tolist())
            rewards[done.cpu().squeeze(-1).numpy() == 1.0] = 0.0

            obs = obs2

        return np.mean(completed_episodes)

    def set_weights(self, weights):
        """
        Update the worker actor version with provided weights.

        weights: dict of tensors
            Dict containing actor weights to be set.
        """
        self.actor_version = weights["version"]
        self.actor.load_state_dict(weights["weights"])

    def update_algo_parameter(self, parameter_name, new_parameter_value):
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
        self.algo.update_algo_parameter(parameter_name, new_parameter_value)

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
        if hasattr(self.storage, parameter_name):
            setattr(self.storage, parameter_name, new_parameter_value)

    def replace_core_component(self, component_name, new_component_factory):
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
        if hasattr(self, component_name):
            if component_name == "algo":
                new_component_component = new_component_factory(
                    self.device, self.actor)
            elif component_name == "envs_train":
                new_component_component = new_component_factory(
                    self.device, self.index_worker)
            elif component_name == "envs_test":
                new_component_component = new_component_factory(
                    self.device, self.index_worker, "test")
            else:
                new_component_component = new_component_factory(self.device)
            setattr(self, component_name, new_component_component)

