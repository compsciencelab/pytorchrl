import os
import ray
import time
import torch
import queue
import threading
from shutil import copy2
from copy import deepcopy

import pytorchrl as prl
from pytorchrl.scheme.base.worker import Worker as W
from pytorchrl.scheme.utils import ray_get_and_free, broadcast_message, pack, unpack

# Puts a limit to the allowed policy lag
max_queue_size = 10


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
    compress_grads_to_send : bool
        Whether or not to compress gradients before sending then to the update worker.
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
    compress_grads_to_send : bool
        Whether or not to compress gradients before sending then to the update worker.
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
                 col_communication=prl.SYNC,
                 compress_grads_to_send=False,
                 col_execution=prl.CENTRAL,
                 col_fraction_workers=1.0,
                 initial_weights=None,
                 device=None):

        self.index_worker = index_worker
        super(GWorker, self).__init__(index_worker)

        # Define counters and other attributes
        self.iter = 0
        self.col_communication = col_communication
        self.compress_grads_to_send = compress_grads_to_send
        self.processing_time_start = None

        # Computation device
        dev = device or "cuda" if torch.cuda.is_available() else "cpu"

        # Create CWorkerSet instance
        self.col_workers = col_workers_factory(dev, initial_weights, index_worker)
        self.local_worker = self.col_workers.local_worker()
        self.remote_workers = self.col_workers.remote_workers()
        self.num_remote_workers = len(self.remote_workers)
        self.num_collection_workers = 1 if self.num_remote_workers == 0 else self.num_remote_workers

        # Get Actor Critic instance
        self.actor = self.local_worker.actor

        # Get Algorithm instance
        self.algo = self.local_worker.algo

        # Get storage instance.
        if col_communication == prl.ASYNC and self.local_worker.envs_train is not None:

            # If async collection, c_worker and g_worker need different storage instances
            # To avoid overwriting data. Define c_worker one with the minimum size required,
            # to save memory
            size1 = self.local_worker.storage.max_size
            size2 = self.local_worker.algo.update_every
            size2 = size2 * 2 if size2 is not None else float("Inf")
            new_size = min(size1, size2)

            if self.local_worker.envs_train is not None:

                # Create a storage class deepcopy, but be careful with envs attribute,
                # can not be deep-copied
                envs = getattr(self.local_worker.storage, "envs")
                self.local_worker.storage.envs = None
                self.storage = deepcopy(self.local_worker.storage)
                self.local_worker.storage.envs = envs
                self.local_worker.update_storage_parameter("max_size", new_size)
            else:
                self.storage = self.local_worker.storage
                for e in self.remote_workers:
                    e.update_storage_parameter.remote("max_size", new_size)
        else:
            self.storage = self.local_worker.storage

        if len(self.remote_workers) > 0 or (len(self.remote_workers) == 0 and
                self.local_worker.envs_train is not None):

            # Create CollectorThread
            self.collector = CollectorThread(
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
        Pulls data from `self.collector.queue`, then perform a gradient computation step.

        Parameters
        ----------
        distribute_gradients : bool
            If True, gradients will be directly shared across remote workers
            and optimization steps will executed in a decentralised way.

        Returns
        -------
        grads : list of tensors
            List of actor gradients.
        info : dict
            Summary dict of relevant gradient operation information.
        """

        self.get_data()
        grads, info = self.get_grads(distribute_gradients)

        if distribute_gradients:
            self.apply_gradients()

        # Encode data if self.compress_data_to_send is True
        data_to_send = pack((grads, info)) if self.compress_grads_to_send else (grads, info)

        return data_to_send

    def get_data(self):
        """
        Pulls data from `self.collector.queue` and prepares batches to
        compute gradients.
        """

        try:  # Try pulling next batch
            self.next_batch = self.batches.__next__()
            # Only use episode information once
            self.info.pop(prl.EPISODES, None)
            # Only record collected samples once
            self.info[prl.NUMSAMPLES] = 0

        # except (StopIteration, AttributeError):
        except Exception:

            # Check if more batches with the same data are required
            can_reuse_data = hasattr(self.algo, "reuse_data")
            if not can_reuse_data or not self.algo.reuse_data:

                # Get new data
                if self.col_communication == prl.SYNC:
                    self.collector.step()
                data, self.info = self.collector.queue.get()

                # Calculate and log data collection-to-processing time ratio
                if not self.processing_time_start:
                    self.processing_time_start = time.time()
                else:
                    self.info[prl.TIME][prl.PROCESSING] = time.time() - self.processing_time_start
                    self.info[prl.TIME][prl.CPRATIO] = (
                        self.info[prl.TIME][prl.COLLECTION] / self.num_collection_workers) / self.info[prl.TIME][
                        prl.PROCESSING]
                    self.processing_time_start = time.time()

                # Proprocess new data
                self.storage.insert_data_slice(data)
                self.storage.before_gradients()
            else:
                # Only record collected samples once
                self.info[prl.NUMSAMPLES] = 0

            # Genarate batches
            self.batches = self.storage.generate_batches(
                self.algo.num_mini_batch,
                self.algo.mini_batch_size,
                self.algo.num_epochs)
            self.next_batch = self.batches.__next__()

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
        grads : list of tensors
            List of actor gradients.
        info : dict
            Summary dict of relevant gradient operation information.
        """

        # Get gradients and algorithm-related information
        t = time.time()
        grads, algo_info = self.compute_gradients(self.next_batch, distribute_gradients)
        compute_time = time.time() - t

        # Add extra information to info dict
        self.info[prl.ALGORITHM] = algo_info
        self.info[prl.VERSION][prl.GRADIENT] = self.local_worker.actor_version
        self.info[prl.TIME][prl.GRADIENT] = compute_time

        # Run after gradients data process (if any)
        info = self.storage.after_gradients(self.next_batch, self.info)

        # Update iteration counter
        self.iter += 1

        return grads, info

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
        grads : list of tensors
            List of actor gradients.
        info : dict
            Summary dict with relevant gradient-related information.
        """

        grads, info = self.algo.compute_gradients(batch, grads_to_cpu=not distribute_gradients)

        if distribute_gradients:

            if torch.cuda.is_available():
                for g in grads:
                    torch.distributed.all_reduce(g, op=torch.distributed.ReduceOp.SUM)
            else:
                torch.distributed.all_reduce_coalesced(grads, op=torch.distributed.ReduceOp.SUM)

            for p in self.actor.parameters():
                if p.grad is not None:
                    p.grad /= self.num_remote_workers

            grads = None

        return grads, info

    def apply_gradients(self, gradients=None):
        """Update Actor Critic model"""

        self.local_worker.actor_version += 1
        self.algo.apply_gradients(gradients)
        if self.col_communication == prl.SYNC and len(self.remote_workers) > 0:
            self.collector.broadcast_new_weights()

    def set_weights(self, actor_weights):
        """
        Update the worker actor version with provided weights.

        weights : dict of tensors
            Dict containing actor weights to be set.
        """

        self.local_worker.actor_version = actor_weights[prl.VERSION]
        self.local_worker.algo.set_weights(actor_weights[prl.WEIGHTS])

    def update_algorithm_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of Worker.algo, change its value to
        `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Algorithm attribute name
        """

        self.local_worker.update_algorithm_parameter(parameter_name, new_parameter_value)
        for e in self.remote_workers:
            e.update_algorithm_parameter.remote(parameter_name, new_parameter_value)

        self.algo.update_algorithm_parameter(parameter_name, new_parameter_value)
        for e in self.col_workers.remote_workers():
            e.update_algorithm_parameter.remote(parameter_name, new_parameter_value)

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
        os.replace(fname + '.tmp', fname)
        save_name = fname + ".{}".format(self.local_worker.actor_version)
        copy2(fname, save_name)
        return save_name

    def stop(self):
        """Stop collecting data."""

        if hasattr(self, "collector"):
            self.collector.stopped = True
        self.local_worker.stop()
        for e in self.remote_workers:
            e.stop.remote()
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
    queue : queue.Queue
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
                 index_worker,
                 local_worker,
                 remote_workers,
                 col_fraction_workers=1.0,
                 col_communication=prl.SYNC,
                 col_execution=prl.CENTRAL,
                 broadcast_interval=1):

        self.index_worker = index_worker
        threading.Thread.__init__(self)

        self.stopped = False
        self.queue = queue.SimpleQueue()
        self.col_execution = col_execution
        self.col_communication = col_communication
        self.broadcast_interval = broadcast_interval
        self.fraction_workers = col_fraction_workers

        self.local_worker = local_worker
        self.remote_workers = remote_workers
        self.num_remote_workers = len(self.remote_workers)
        self.num_collection_workers = 1 if self.num_remote_workers == 0 else self.num_remote_workers

        # Counters
        self.num_sent_since_broadcast = 0

        if col_execution == prl.CENTRAL and col_communication == prl.SYNC:
            pass

        elif col_execution == prl.CENTRAL and col_communication == prl.ASYNC:
            if self.local_worker.envs_train: self.start()  # Start CollectorThread

        elif col_execution == prl.PARALLEL and col_communication == prl.SYNC:
            pass

        elif col_execution == prl.PARALLEL and col_communication == prl.ASYNC:
            self.pending_tasks = {}
            self.broadcast_new_weights()
            for w in self.remote_workers:
                    future = w.collect_data.remote()
                    self.pending_tasks[future] = w
            self.start()

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

        if self.col_execution == prl.CENTRAL and self.col_communication == prl.SYNC:

            rollouts = self.local_worker.collect_data(listen_to=["sync"], data_to_cpu=False)
            rollouts = unpack(rollouts) if type(rollouts) == str else rollouts
            self.queue.put(rollouts)
            while self.queue.qsize() >= max_queue_size:
                time.sleep(0.5)

        elif self.col_execution == prl.CENTRAL and self.col_communication == prl.ASYNC:

            rollouts = self.local_worker.collect_data(data_to_cpu=False)
            rollouts = unpack(rollouts) if type(rollouts) == str else rollouts
            self.queue.put(rollouts)
            while self.queue.qsize() >= max_queue_size:
                time.sleep(0.5)

        elif self.col_execution == prl.PARALLEL and self.col_communication == prl.SYNC:

            # Start data collection in all workers
            worker_key = "worker_{}".format(self.index_worker)
            broadcast_message(worker_key, b"start-continue")

            pending_samples = [e.collect_data.remote(
                listen_to=["sync", worker_key]) for e in self.remote_workers]

            if self.fraction_workers < 1.0:
                # Keep checking how many workers have finished until
                # percent% are ready
                samples_ready = []
                while len(samples_ready) < (self.num_remote_workers * self.fraction_workers):
                    samples_ready, samples_not_ready = ray.wait(
                        pending_samples, num_returns=len(pending_samples), timeout=0.001)

                # Send stop message to the workers
                broadcast_message(worker_key, b"stop")

            # Compute model updates
            for r in pending_samples:
                rollouts = ray_get_and_free(r)
                rollouts = unpack(rollouts) if type(rollouts) == str else rollouts
                self.queue.put(rollouts)
                while self.queue.qsize() >= max_queue_size:
                    time.sleep(0.5)

        elif self.col_execution == prl.PARALLEL and self.col_communication == prl.ASYNC:

            # Wait for first worker to finish
            assert len(list(self.pending_tasks.keys())) == len(self.remote_workers)

            wait_results = ray.wait(list(self.pending_tasks.keys()))
            future = wait_results[0][0]
            w = self.pending_tasks.pop(future)

            # Retrieve rollouts and add them to queue
            rollouts = ray_get_and_free(future)
            rollouts = unpack(rollouts) if type(rollouts) == str else rollouts
            self.queue.put(rollouts)

            while self.queue.qsize() >= max_queue_size:
                time.sleep(0.5)

            # Schedule a new collection task
            future = w.collect_data.remote()
            self.pending_tasks[future] = w

    def should_broadcast(self):
        """Returns whether broadcast() should be called to update weights."""
        return self.num_sent_since_broadcast >= self.broadcast_interval

    def broadcast_new_weights(self):
        """Broadcast a new set of weights from the local worker."""
        if self.num_remote_workers > 0:
            latest_weights = ray.put({
                prl.VERSION: self.local_worker.actor_version,
                prl.WEIGHTS: self.local_worker.get_weights()})
            for e in self.remote_workers:
                e.set_weights.remote(latest_weights)
            self.num_sent_since_broadcast = 0


