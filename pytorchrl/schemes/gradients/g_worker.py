import os
import ray
import time
import torch
import threading
from shutil import copy2
from copy import deepcopy
from six.moves import queue
from functools import partial
from collections import defaultdict, deque

from ..base.worker import Worker as W
from ..utils import ray_get_and_free, broadcast_message


class GWorker(W):
    """
    Worker class handling gradient computation.

    This class wraps an actor instance, a storage class instance and a
    worker set of remote data collection workers. It receives data from the
    collection workers and computes gradients following a logic defined in
    function self.step(), which will be called from the Learner class.

    Parameters
    ----------
    index_worker : int
        Worker index.
    col_workers_factory : func
        A function that creates a set of data collection workers.
    col_communication : str
        Communication coordination pattern for data collection.
    col_execution : str
        Execution patterns for data collection.
    col_fraction_workers : float
        Minimum fraction of samples required to stop if collection is
        synchronously coordinated and most workers have finished their
        collection task.
    device : str
        "cpu" or specific GPU "cuda:number`" to use for computation.
    initial_weights : ray object ID
        Initial model weights.

    Attributes
    ----------
    index_worker : int
        Index assigned to this worker.
    iter : int
        Number of times gradients have been computed and sent.
    col_communication : str
        Communication coordination pattern for data collection.
    col_workers : CWorkerSet
        A CWorkerSet class instance.
    local_worker : CWorker
        col_workers local worker.
    remote_workers : List of CWorker's
        col_workers remote data collection workers.
    num_workers : int
        Number of collection remote workers.
    actor : Actor
        An actor class instance.
    algo : Algo
        An algorithm class instance.
    storage : Storage
        A Storage class instance.
    inqueue : queue.Queue
        Input queue where incoming collected samples are placed.
    collector : CollectorThread
        Class handling data collection via col_workers and placing incoming
        rollouts into the input queue `inqueue`.
    """

    def __init__(self,
                 index_worker,
                 col_workers_factory,
                 col_communication="synchronous",
                 col_execution="distributed",
                 col_fraction_workers=1.0,
                 initial_weights=None,
                 device=None):

        super(GWorker, self).__init__(index_worker)

        # Define counters and other attributes
        self.iter = 0
        self.col_communication = col_communication

        # Computation device
        dev = device or "cuda" if torch.cuda.is_available() else "cpu"

        # Create CWorkerSet instance
        self.col_workers = col_workers_factory(dev, initial_weights, index_worker)
        self.local_worker = self.col_workers.local_worker()
        self.remote_workers = self.col_workers.remote_workers()
        self.num_workers = len(self.remote_workers)

        # Get Actor Critic instance
        self.actor = self.local_worker.actor

        # Get Algorithm instance
        self.algo = self.local_worker.algo

        # Get storage instance
        if col_communication == "synchronous":
            self.storage = self.local_worker.storage
        else:
            self.storage = deepcopy(self.local_worker.storage)

        # Queue
        self.inqueue = queue.Queue(maxsize=100)

        # Create CollectorThread
        self.collector = CollectorThread(
            input_queue=self.inqueue,
            index_worker=index_worker,
            local_worker=self.local_worker,
            remote_workers=self.remote_workers,
            col_communication=col_communication,
            col_fraction_workers=col_fraction_workers,
            col_execution=col_execution,
            broadcast_interval=1)

        # Print worker information
        self.print_worker_info()

    @property
    def actor_version(self):
        """Number of times Actor has been updated."""
        return self.local_worker.actor_version

    def step(self, distribute_gradients=False):
        """
        Pulls data from `self.inqueue`, then perform a gradient computation step.

        Parameters
        ----------
        distribute_gradients : bool
            If True, gradients will be directly shared across remote workers
            and optimization steps will executed in a decentralised way.

        Returns
        -------
        grads: list of tensors
            List of actor gradients.
        info : dict
            Summary dict of relevant gradient operation information.
        """
        self.get_data()
        grads, info = self.get_grads(distribute_gradients)
        if distribute_gradients: self.apply_gradients()
        return grads, info

    def get_grads(self, distribute_gradients=False):
        """
        Perform a gradient computation step.

        Parameters
        ----------
        distribute_gradients : bool
            If True, gradients will be directly shared across remote workers
            and optimization steps will executed in a decentralised way.

        Returns
        -------
        grads: list of tensors
            List of actor gradients.
        info : dict
            Summary dict of relevant gradient operation information.
        """

        # Collect data and prepare data batches
        if self.iter % (self.algo.num_epochs * self.algo.num_mini_batch) == 0:

            self.storage.add_data(self.data)
            self.storage.before_update(self.actor, self.algo)
            self.batches = self.storage.generate_batches(
                self.algo.num_mini_batch, self.algo.mini_batch_size,
                self.algo.num_epochs, self.actor.is_recurrent)

        # Compute gradients, get algo info
        grads, info = self.compute_gradients(
            self.batches.__next__(), distribute_gradients)

        # Add extra information to info dict
        info.update(self.col_info)
        self.col_info.update({"collected_samples": 0})
        info.update({"grad_version": self.local_worker.actor_version})

        self.iter += 1

        return grads, info

    def get_data(self):
        """Pulls data from `self.inqueue`"""
        if self.iter % (self.algo.num_epochs * self.algo.num_mini_batch) == 0:
            if self.col_communication == "synchronous": self.collector.step()
            self.data, self.col_info = self.inqueue.get(timeout=300)

    def compute_gradients(self, batch, distribute_gradients):
        """
        Calculate actor gradients and update networks.

        Parameters
        ----------
        batch : dict
            data batch containing all required tensors to compute algo loss.
        distribute_gradients : bool
            If True, gradients will be directly shared across remote workers
            and optimization steps will executed in a decentralised way.

        Returns
        -------
        grads: list of tensors
            List of actor gradients.
        info : dict
            Summary dict with relevant gradient-related information.
        """

        t = time.time()
        grads, info = self.algo.compute_gradients(batch, grads_to_cpu=not distribute_gradients)
        compute_time = time.time() - t
        info.update({"time/compute_grads": compute_time})

        if distribute_gradients:

            t = time.time()
            if torch.cuda.is_available():
                for g in grads:
                    torch.distributed.all_reduce(g, op=torch.distributed.ReduceOp.SUM)
            else:
                torch.distributed.all_reduce_coalesced(grads, op=torch.distributed.ReduceOp.SUM)

            for p in self.actor.parameters():
                if p.grad is not None:
                    p.grad /= self.num_workers

            avg_grads_t = time.time() - t
            grads = None

            info.update({"time/avg_grads": avg_grads_t})

        return grads, info

    def apply_gradients(self, gradients=None):
        """Update Actor Critic model"""
        self.local_worker.actor_version += 1
        self.algo.apply_gradients(gradients)
        if self.col_communication == "synchronous":
            self.collector.broadcast_new_weights()

    def set_weights(self, weights):
        """
        Update the worker actor version with provided weights.

        weights: dict of tensors
            Dict containing actor weights to be set.
        """
        self.local_worker.actor_version = weights["version"]
        self.local_worker.algo.set_weights(weights["weights"])

    def update_algo_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of Worker.algo, change its value to
        `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Algorithm attribute name
        """
        self.local_worker.update_algo_parameter(parameter_name, new_parameter_value)
        for e in self.remote_workers:
            e.update_algo_parameter.remote(parameter_name, new_parameter_value)


        self.algo.update_algo_parameter(parameter_name, new_parameter_value)
        for e in self.col_workers.remote_workers():
            e.update_algo_parameter.remote(parameter_name, new_parameter_value)

    def save_model(self, fname):
        """
        Save current version of actor as a torch loadable checkpoint.

        Parameters
        ----------
        fname : str
            Filename given to the checkpoint.

        Returns
        -------
        save_name : str
            Path to saved file.
        """
        torch.save(self.local_worker.actor.state_dict(), fname + ".tmp")
        os.rename(fname + '.tmp', fname)
        save_name = fname + ".{}".format(self.local_worker.actor_version)
        copy2(fname, save_name)
        return save_name

    def stop(self):
        """Stop collecting data."""
        self.collector.stopped = True
        for e in self.collector.remote_workers:
            e.terminate_worker.remote()


class CollectorThread(threading.Thread):
    """
    This class receives data samples from the data collection workers and
    queues them into the data input_queue.

    Parameters
    ----------
    index_worker : int
        Index assigned to this worker.
    input_queue : queue.Queue
        Queue to store the data dicts received from data collection workers.
    local_worker : Worker
        Local worker that acts as a parameter server.
    remote_workers : list of Workers
        Set of workers collecting and sending rollouts.
    col_fraction_workers : float
        Minimum fraction of samples required to stop if collection is
        synchronously coordinated and most workers have finished their
        collection task.
    col_communication : str
        Communication coordination pattern for data collection.
    col_execution : str
        Execution patterns for data collection.
    broadcast_interval : int
        After how many central updates, model weights should be broadcasted to
        remote collection workers.

    Attributes
    ----------
    stopped : bool
        Whether or not the thread in running.
    inqueue : queue.Queue
        Queue to store the data dicts received from data collection workers.
    index_worker : int
        Index assigned to this worker.
    local_worker : CWorker
        col_workers local worker.
    remote_workers : List of CWorker's
        col_workers remote data collection workers.
    num_workers : int
        Number of collection remote workers.
    broadcast_interval : int
        After how many collection step model weights should be broadcasted to
        remote collection workers.
    num_sent_since_broadcast : int
        Number of data dicts received since last model weights were broadcasted.
    """

    def __init__(self,
                 input_queue,
                 index_worker,
                 local_worker,
                 remote_workers,
                 col_fraction_workers=1.0,
                 col_communication="synchronous",
                 col_execution="distributed",
                 broadcast_interval=1):

        threading.Thread.__init__(self)

        self.stopped = False
        self.inqueue = input_queue
        self.index_worker = index_worker
        self.col_execution = col_execution
        self.col_communication = col_communication
        self.broadcast_interval = broadcast_interval
        self.fraction_workers = col_fraction_workers

        self.local_worker = local_worker
        self.remote_workers = remote_workers
        self.num_workers = len(self.remote_workers)

        # Counters and metrics
        self.num_sent_since_broadcast = 0
        self.metrics = defaultdict(partial(deque, maxlen=100))

        if col_execution == "centralised" and col_communication == "synchronous":
            pass

        elif col_execution == "centralised" and col_communication == "asynchronous":
            # Start CollectorThread
            self.start()

        elif col_execution == "decentralised" and col_communication == "synchronous":
            pass

        elif col_execution == "decentralised" and col_communication == "asynchronous":
            # Start CollectorThread
            self.pending_tasks = {}
            self.broadcast_new_weights()
            for w in self.remote_workers:
                for _ in range(2):
                    future = w.collect_data.remote()
                    self.pending_tasks[future] = w
            self.start()

        else:
            raise NotImplementedError


    def run(self):
        while not self.stopped:
            # First, collect data
            self.step()
            # Then, update counter and broadcast weights to worker if necessary
            self.num_sent_since_broadcast += 1
            if self.should_broadcast():
                self.broadcast_new_weights()

    def step(self):
        """
        Collects data from remote workers and puts it in the GWorker queue.
        """

        if self.col_execution == "centralised" and self.col_communication == "synchronous":

            rollouts = self.local_worker.collect_data(listen_to=["sync"], data_to_cpu=False)
            self.inqueue.put(rollouts)

        elif self.col_execution == "centralised" and self.col_communication == "asynchronous":
            rollouts = self.local_worker.collect_data(data_to_cpu=False)
            self.inqueue.put(rollouts)

        elif self.col_execution == "decentralised" and self.col_communication == "synchronous":

            # Start data collection in all workers
            worker_key = "worker_{}".format(self.index_worker)
            broadcast_message(worker_key, b"start-continue")

            pending_samples = [e.collect_data.remote(
                listen_to=["sync", worker_key]) for e in self.remote_workers]

            # Keep checking how many workers have finished until percent% are ready
            samples_ready, samples_not_ready = ray.wait(
                pending_samples, num_returns=len(pending_samples), timeout=0.5)
            while len(samples_ready) < (self.num_workers * self.fraction_workers):
                samples_ready, samples_not_ready = ray.wait(
                    pending_samples, num_returns=len(pending_samples), timeout=0.5)

            # Send stop message to the workers
            broadcast_message(worker_key, b"stop")

            # Compute model updates
            for r in pending_samples: self.inqueue.put(ray_get_and_free(r))


        elif self.col_execution == "decentralised" and self.col_communication == "asynchronous":

            # Wait for first worker to finish
            wait_results = ray.wait(list(self.pending_tasks.keys()))
            future = wait_results[0][0]
            w = self.pending_tasks.pop(future)

            # Retrieve rollouts and add them to queue
            self.inqueue.put(ray_get_and_free(future))

            # Then, update counter and broadcast weights to worker if necessary
            self.num_sent_since_broadcast += 1
            if self.should_broadcast():
                self.broadcast_new_weights()

            # Schedule a new collection task
            future = w.collect_data.remote()
            self.pending_tasks[future] = w

        else:
            raise NotImplementedError

    def should_broadcast(self):
        """Returns whether broadcast() should be called to update weights."""
        return self.num_sent_since_broadcast >= self.broadcast_interval

    def broadcast_new_weights(self):
        """Broadcast a new set of weights from the local worker."""
        if self.num_workers > 0:
            latest_weights = ray.put({
                "version": self.local_worker.actor_version,
                "weights": self.local_worker.get_weights()})
            for e in self.remote_workers:
                e.set_weights.remote(latest_weights)
            self.num_sent_since_broadcast = 0


