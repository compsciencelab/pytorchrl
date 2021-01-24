import ray
from .worker import default_remote_config


class WorkerSet:
    """
    Class to better handle the operations of ensembles of Workers.
    Contains common functionality across all worker sets.

    Parameters
    ----------
    worker : func
        A function that creates a worker class.
    worker_params : dict
        Worker class kwargs.
    worker_remote_config : dict
        Ray resource specs for the remote workers.
    num_workers : int
        Num workers replicas in the worker_set.
    add_local_worker : bool
        Whether or not to include have a non-remote worker in the worker set.

    Attributes
    ----------
    worker_class : python class
        Worker class to be instantiated to create Ray remote actors.
    remote_config : dict
        Ray resource specs for the remote workers.
    worker_params : dict
        Keyword arguments of the worker_class.
    num_workers : int
        Number of remote workers in the worker set.
    """

    def __init__(self,
                 worker,
                 worker_params,
                 index_parent_worker,
                 worker_remote_config=default_remote_config,
                 num_workers=1,
                 local_device=None,
                 initial_weights=None,
                 add_local_worker=True,
                 total_parent_workers=None):

        self.worker_class = worker
        self.num_workers = num_workers
        self.worker_params = worker_params
        self.remote_config = worker_remote_config

        if add_local_worker:

            local_params = worker_params.copy()
            local_params.update(
                {"device": local_device, "initial_weights": initial_weights})

            # If multiple grad workers the collection workers of grad worker with index 0 should not collect
            if worker.__name__ == "CWorker" and total_parent_workers > 0 and index_parent_worker == 0:
                self.num_workers = 0
                _ = local_params.pop("test_envs_factory")
                _ = local_params.pop("train_envs_factory")

            # If multiple col workers, local collection workers don't need to collect
            elif worker.__name__ == "CWorker" and num_workers > 0:
                _ = local_params.pop("test_envs_factory")
                _ = local_params.pop("train_envs_factory")

            self._local_worker = self._make_worker(
                self.worker_class, index_worker=0,
                worker_params=local_params)

        else:
            self._local_worker = None

        self._remote_workers = []
        if self.num_workers > 0:
            self.add_workers(self.num_workers)

    @staticmethod
    def _make_worker(cls, index_worker, worker_params):
        """
        Create a single worker.

        Parameters
        ----------
        index_worker : int
            Index assigned to remote worker.
        worker_params : dict
            Keyword parameters of the worker_class.

        Returns
        -------
        w : python class
            An instance of worker class cls
        """
        w = cls(index_worker=index_worker, **worker_params)
        return w

    def add_workers(self, num_workers):
        """
        Create and add a number of remote workers to this worker set.

        Parameters
        ----------
        num_workers : int
            Number of remote workers to create.
        """
        self.worker_params.update({"initial_weights": ray.put(
            {"version": 0, "weights": self._local_worker.get_weights()})})
        cls = self.worker_class.as_remote(**self.remote_config).remote
        self._remote_workers.extend([
            self._make_worker(cls, index_worker=i + 1, worker_params=self.worker_params)
            for i in range(num_workers)])

    def local_worker(self):
        """Return local worker"""
        return self._local_worker

    def remote_workers(self):
        """Returns list of remote workers"""
        return self._remote_workers

    def stop(self):
        """Stop all remote workers"""
        for w in self.remote_workers():
            w.__ray_terminate__.remote()

