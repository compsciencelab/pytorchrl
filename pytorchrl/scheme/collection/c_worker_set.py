from .c_worker import CWorker
from ..base.worker_set import WorkerSet as WS
from ..base.worker import default_remote_config


class CWorkerSet(WS):
    """
    Class to better handle the operations of ensembles of CWorkers.

    Parameters
    ----------
    num_workers : int
        Number of remote workers in the worker set.
    algo_factory : func
        A function that creates an algorithm class.
    actor_factory : func
        A function that creates a policy.
    storage_factory : func
        A function that create a rollouts storage.
    train_envs_factory : func
        A function to create train environments.
    local_device : str
        "cpu" or specific GPU "cuda:number`" to use for computation.
    initial_weights : ray object ID
        Initial model weights.
    fraction_samples :
        Minimum fraction of samples required to stop if collection is
        synchronously coordinated and most workers have finished their
        collection task.
    test_envs_factory : func
        A function to create test environments.
    worker_remote_config : dict
        Ray resource specs for the remote workers.

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
                 num_workers,
                 index_parent,
                 algo_factory,
                 actor_factory,
                 storage_factory,
                 local_device=None,
                 initial_weights=None,
                 fraction_samples=1.0,
                 total_parent_workers=0,
                 train_envs_factory=lambda x, y, z: None,
                 test_envs_factory=lambda v, x, y, c: None,
                 worker_remote_config=default_remote_config):

        self.worker_class = CWorker
        default_remote_config.update(worker_remote_config)
        self.remote_config = default_remote_config
        self.worker_params = {
            "index_parent": index_parent,
            "algo_factory": algo_factory,
            "storage_factory": storage_factory,
            "test_envs_factory": test_envs_factory,
            "train_envs_factory": train_envs_factory,
            "fraction_samples": fraction_samples,
            "actor_factory": actor_factory,
        }

        self.num_workers = num_workers
        super(CWorkerSet, self).__init__(
            worker=self.worker_class,
            local_device=local_device,
            num_workers=self.num_workers,
            initial_weights=initial_weights,
            worker_params=self.worker_params,
            index_parent_worker=index_parent,
            worker_remote_config=self.remote_config,
            total_parent_workers=total_parent_workers)

    @classmethod
    def create_factory(cls,
                       num_workers,
                       algo_factory,
                       actor_factory,
                       storage_factory,
                       test_envs_factory,
                       train_envs_factory,
                       total_parent_workers=0,
                       col_fraction_samples=1.0,
                       col_worker_resources=default_remote_config):
        """
        Returns a function to create new CWorkerSet instances.

        Parameters
        ----------
        num_workers : int
            Number of remote workers in the worker set.
        algo_factory : func
            A function that creates an algorithm class.
        actor_factory : func
            A function that creates a policy.
        storage_factory : func
            A function that create a rollouts storage.
        train_envs_factory : func
            A function to create train environments.
        col_fraction_samples :
            Minimum fraction of samples required to stop if collection is
            synchronously coordinated and most workers have finished their
            collection task.
        test_envs_factory : func
            A function to create test environments.
        col_worker_resources : dict
            Ray resource specs for the remote workers.

        Returns
        -------
        collection_worker_set_factory : func
            creates a new CWorkerSet class instance.
        """

        def collection_worker_set_factory(device, initial_weights, index_parent):
            """
            Creates and returns a CWorkerSet class instance.

            Parameters
            ----------
            device : str
                "cpu" or specific GPU "cuda:number`" to use for computation.
            initial_weights : ray object ID
                Initial model weights.

            Returns
            -------
            CWorkerSet : CWorkerSet
                A new CWorkerSet class instance.
            """
            return cls(
                local_device=device,
                num_workers=num_workers,
                index_parent=index_parent,
                algo_factory=algo_factory,
                actor_factory=actor_factory,
                storage_factory=storage_factory,
                initial_weights=initial_weights,
                fraction_samples=col_fraction_samples,
                test_envs_factory=test_envs_factory,
                train_envs_factory=train_envs_factory,
                total_parent_workers=total_parent_workers,
                worker_remote_config=col_worker_resources)

        return collection_worker_set_factory
