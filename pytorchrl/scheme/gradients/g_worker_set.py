from .g_worker import GWorker
from ..base.worker_set import WorkerSet as WS
from ..base.worker import default_remote_config


class GWorkerSet(WS):
    """
    Class to better handle the operations of ensembles of GWorkers.

    Parameters
    ----------
    num_workers : int
        Number of remote workers in the worker set.
    local_device : str
        "cpu" or specific GPU "cuda:number`" to use for computation.
    col_execution : str
        Execution patterns for data collection.
    col_communication : str
        Communication coordination pattern for data collection.
    col_workers_factory : func
        A function that creates a set of data collection workers.
    col_fraction_workers : float
        Minimum fraction of samples required to stop if collection is
        synchronously coordinated and most workers have finished their
        collection task.
    grad_worker_resources : dict
        Ray resource specs for the remote workers.
    initial_weights : ray object ID
        Initial model weights.

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
                 local_device,
                 col_execution,
                 col_communication,
                 col_workers_factory,
                 col_fraction_workers,
                 grad_worker_resources):

        self.worker_class = GWorker
        self.num_workers = num_workers
        default_remote_config.update(grad_worker_resources)
        self.remote_config = default_remote_config

        self.worker_params = {
            "col_execution": col_execution,
            "col_communication": col_communication,
            "col_workers_factory": col_workers_factory,
            "col_fraction_workers": col_fraction_workers,
        }

        super(GWorkerSet, self).__init__(
            worker=self.worker_class,
            local_device=local_device,
            num_workers=self.num_workers,
            worker_params=self.worker_params,
            index_parent_worker=index_parent,
            worker_remote_config=self.remote_config)

    @classmethod
    def create_factory(cls,
                       num_workers,
                       col_workers_factory,
                       col_fraction_workers=1.0,
                       col_execution="distributed",
                       col_communication="synchronous",
                       grad_worker_resources=default_remote_config):
        """
        Returns a function to create new CWorkerSet instances.

        Parameters
        ----------
        num_workers : int
            Number of remote workers in the worker set.
        col_execution : str
            Execution patterns for data collection.
        col_communication : str
            Communication coordination pattern for data collection.
        col_workers_factory : func
            A function that creates a set of data collection workers.
        col_fraction_workers : float
            Minimum fraction of samples required to stop if collection is
            synchronously coordinated and most workers have finished their
            collection task.
        grad_worker_resources : dict
            Ray resource specs for the remote workers.

        Returns
        -------
        grad_worker_set_factory : func
            creates a new GWorkerSet class instance.
        """
        def grad_worker_set_factory(device, index_parent):
            """
            Creates and returns a GWorkerSet class instance.

            Parameters
            ----------
            device : str
                "cpu" or specific GPU "cuda:number`" to use for computation.

            Returns
            -------
            GWorkerSet : GWorkerSet
                A new GWorkerSet class instance.
            """
            return cls(
                local_device=device,
                num_workers=num_workers,
                index_parent=index_parent,
                col_execution=col_execution,
                col_communication=col_communication,
                col_workers_factory=col_workers_factory,
                col_fraction_workers=col_fraction_workers,
                grad_worker_resources=grad_worker_resources)

        return grad_worker_set_factory
