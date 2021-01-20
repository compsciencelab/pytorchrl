from .collection.c_worker_set import CWorkerSet
from .gradients.g_worker_set import GWorkerSet
from .updates.u_worker import UWorker


class Scheme:
    """
    Class to define training schemes and handle creation and operation
    of its workers.

    Parameters
    ----------
    algo_factory : func
        A function that creates an algorithm class.
    actor_factory : func
        A function that creates a policy.
    storage_factory : func
        A function that create a rollouts storage.
    train_envs_factory : func
        A function to create train environments.
    test_envs_factory : func
        A function to create test environments.
    col_remote_workers : int
        Number of data collection workers per gradient worker.
    col_communication : str
        Communication coordination pattern for data collection.
    col_worker_resources : dict
        Ray resource specs for collection remote workers.
    sync_col_specs : dict
        specs about minimum fraction_samples [0 - 1.0] and minimum
        fraction_workers [0 - 1.0] required in synchronous data collection.
    grad_remote_workers : int
        Number of gradient workers.
    grad_communication : str
        Communication coordination pattern for gradient computation workers.
    grad_worker_resources : dict
        Ray resource specs for gradient remote workers.
    local_device : str
        "cpu" or specific GPU "cuda:`number`" to use for computation.
    update_execution : str
        Execution patterns for update steps.
    """
    def __init__(self,

                 # core
                 algo_factory,
                 actor_factory,
                 storage_factory,
                 train_envs_factory,
                 test_envs_factory=lambda v, x, y, z: None,

                 # collection
                 num_col_workers=1,
                 col_communication="synchronous",
                 col_worker_resources={"num_cpus": 1, "num_gpus": 0.25},
                 sync_col_specs={"fraction_samples": 1.0, "fraction_workers": 1.0},

                 # gradients
                 num_grad_workers=1,
                 grad_communication="synchronous",
                 grad_worker_resources={"num_cpus": 1, "num_gpus": 0.25},

                 # update
                 local_device=None,
                 update_execution="centralised", # OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main.py

                 ):

        col_execution="parallelised" if num_col_workers > 1 else "centralised"
        grad_execution="parallelised" if num_grad_workers > 1 else "centralised"

        col_workers_factory = CWorkerSet.create_factory(

            # core modules
            algo_factory=algo_factory,
            actor_factory=actor_factory,
            storage_factory=storage_factory,
            test_envs_factory=test_envs_factory,
            train_envs_factory=train_envs_factory,

            # col specs
            num_workers=num_col_workers - 1 if num_col_workers == 1 else num_col_workers,
            col_worker_resources=col_worker_resources,
            col_fraction_samples=sync_col_specs.get("fraction_samples"),

            # grad specs
            total_parent_workers=num_grad_workers - 1 if num_grad_workers == 1 else num_grad_workers,
        )

        grad_workers_factory = GWorkerSet.create_factory(

            # col specs
            col_execution=col_execution,
            col_communication=col_communication,
            col_workers_factory=col_workers_factory,
            col_fraction_workers=sync_col_specs.get("fraction_workers"),

            # grad_specs
            num_workers=num_grad_workers - 1 if num_grad_workers == 1 else num_grad_workers,
            grad_worker_resources=grad_worker_resources,
        )

        self._update_worker = UWorker(

           # col specs
            col_fraction_workers=sync_col_specs.get("fraction_workers"),

            # grad specs
            grad_execution=grad_execution,
            grad_communication=grad_communication,
            grad_workers_factory=grad_workers_factory,

            # update specs
            local_device=local_device,
            update_execution=update_execution,
        )

    def update_worker(self):
        """Return local worker"""
        return self._update_worker