# Data collector
col_workers_factory = CWorkerSet.create_factory(
    algo_factory=algo_factory,
    actor_factory=actor_factory,
    storage_factory=storage_factory,
    test_envs_factory=test_envs_factory,
    train_envs_factory=train_envs_factory,
    num_workers=1,
    col_worker_resources={"num_cpus": 1, "num_gpus": 0.25},
)

# Gradient Collector (has copies of data collectors)
grad_workers_factory = GWorkerSet.create_factory(
    col_communication=prl.SYNC,
    col_workers_factory=col_workers_factory,
    num_workers=1,  # num grad workers in the set
    grad_worker_resources={"num_cpus": 1, "num_gpus": 0.25},
)

# Model Updater (has copies of gradient collectors)
update_worker = UWorker(
    grad_communication=prl.SYNC,
    grad_workers_factory=grad_workers_factory,
    decentralized_update_execution=False,  # Gradients are applied in the update workers (central update)
)


########################################################################################################################

from pytorchrl.scheme import CWorkerSet, GWorkerSet, UWorker
# Collector workers
col_workers_factory = CWorkerSet.create_factory(
    num_workers=1,
    algo_factory=algo_factory,
    actor_factory=actor_factory,
    storage_factory=storage_factory,
    test_envs_factory=test_envs_factory,
    train_envs_factory=train_envs_factory,
    col_worker_resources={"num_cpus": 1, "num_gpus": 0.25},
)
# Gradient Workers
grad_workers_factory = GWorkerSet.create_factory(
    num_workers=1,
    col_communication=prl.SYNC,
    col_workers_factory=col_workers_factory,
    grad_worker_resources={"num_cpus": 1, "num_gpus": 0.25},
)
# Model Updater
update_worker = UWorker(
    grad_communication=prl.SYNC,
    grad_workers_factory=grad_workers_factory,
)
collected_steps = 0
while collected_steps < target_steps:
    # Collect data, take one grad step, update model parameters
    info = update_worker.step()
