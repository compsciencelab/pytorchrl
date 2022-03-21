import logging
logger = logging.getLogger(__name__)

import pytorchrl as prl
from pytorchrl.scheme.collection.c_worker_set import CWorkerSet
from pytorchrl.scheme.gradients.g_worker_set import GWorkerSet
from pytorchrl.scheme.updates.u_worker import UWorker


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
    num_col_workers : int
        Number of data collection workers per gradient worker.
    col_workers_communication : str
        Communication coordination pattern for data collection.
    col_workers_resources : dict
        Ray resource specs for collection remote workers.
    col_preemption_thresholds : dict
        specs about minimum fraction_samples [0 - 1.0] and minimum
        fraction_workers [0 - 1.0] required in synchronous data collection.
    num_grad_workers : int
        Number of gradient workers.
    grad_workers_communication : str
        Communication coordination pattern for gradient computation workers.
    grad_workers_resources : dict
        Ray resource specs for gradient remote workers.
    local_device : str
        "cpu" or specific GPU "cuda:`number`" to use for computation.
    decentralized_update_execution : bool
        Whether the gradients are applied in the update workers (central update)
        or broadcasted to all gradient workers for a decentralized update.
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
                 col_compress_data=False,
                 col_workers_communication=prl.SYNC,
                 col_workers_resources={"num_cpus": 1, "num_gpus": 0.5},
                 col_preemption_thresholds={"fraction_samples": 1.0, "fraction_workers": 1.0},

                 # gradients
                 num_grad_workers=1,
                 grad_compress_data=False,
                 grad_workers_communication=prl.SYNC,
                 grad_workers_resources={"num_cpus": 1, "num_gpus": 0.5},

                 # update
                 local_device=None,
                 decentralized_update_execution=False,  # OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main.py
                 ):

        assert col_workers_communication in (prl.SYNC, prl.ASYNC),\
            "col_workers_communication can only be `prl.SYNC` or `prl.ASYNC`"

        assert grad_workers_communication in (prl.SYNC, prl.ASYNC),\
            "grad_workers_communication can only be `prl.SYNC` or `prl.ASYNC`"

        col_execution = prl.PARALLEL if num_col_workers > 1 else prl.CENTRAL
        grad_execution = prl.PARALLEL if num_grad_workers > 1 else prl.CENTRAL

        col_workers_factory = CWorkerSet.create_factory(

            # core modules
            algo_factory=algo_factory,
            actor_factory=actor_factory,
            storage_factory=storage_factory,
            test_envs_factory=test_envs_factory,
            train_envs_factory=train_envs_factory,

            # col specs
            num_workers=num_col_workers - 1 if num_col_workers == 1 else num_col_workers,
            col_worker_resources=col_workers_resources,
            col_fraction_samples=col_preemption_thresholds.get("fraction_samples"),
            compress_data_to_send=col_compress_data,

            # grad specs
            total_parent_workers=num_grad_workers - 1 if num_grad_workers == 1 else num_grad_workers,
        )

        grad_workers_factory = GWorkerSet.create_factory(

            # col specs
            col_execution=col_execution,
            col_communication=col_workers_communication,
            col_workers_factory=col_workers_factory,
            col_fraction_workers=col_preemption_thresholds.get("fraction_workers"),

            # grad_specs
            num_workers=num_grad_workers - 1 if num_grad_workers == 1 else num_grad_workers,
            grad_worker_resources=grad_workers_resources,
            compress_grads_to_send=grad_compress_data,
        )

        self._update_worker = UWorker(

           # col specs
            col_fraction_workers=col_preemption_thresholds.get("fraction_workers"),

            # grad specs
            grad_execution=grad_execution,
            grad_communication=grad_workers_communication,
            grad_workers_factory=grad_workers_factory,

            # update specs
            local_device=local_device,
            decentralized_update_execution=decentralized_update_execution,
        )

        logger.warning("Created training scheme.")

    def update_worker(self):
        """Return local worker"""
        return self._update_worker

    def get_agent_components(self):
        return {
            "Actor": self._update_worker.local_worker.local_worker.actor.__class__.__name__,
            "Algorithm": self._update_worker.local_worker.local_worker.algo.__class__.__name__,
            "Storage": self._update_worker.local_worker.local_worker.storage.__class__.__name__,
        }
