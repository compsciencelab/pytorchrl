# Data collector
col_workers_factory = CWorkerSet.create_factory(
    algo_factory=algo_factory,
    actor_factory=actor_factory,
    storage_factory=storage_factory,
    test_envs_factory=test_envs_factory,
    train_envs_factory=train_envs_factory,
    num_workers=1,
#    col_fraction_samples=1.0,  # Collect 100% of the samples, no preemption mechanism
    col_worker_resources={"num_cpus": 1, "num_gpus": 0.25},
    total_parent_workers=1,  # TODO: this can be a flag called differntly
)

# Gradient Collector (has copies of data collectors)
grad_workers_factory = GWorkerSet.create_factory(
    col_execution=prl.CENTRAL,  # TODO: remove! this can be known from len(collection_workers)
    col_communication=prl.SYNC,
    col_workers_factory=col_workers_factory,
    num_workers=1,  # num grad workers in the set
#    col_fraction_workers=1.0,
    grad_worker_resources={"num_cpus": 1, "num_gpus": 0.25},
)

# Model Updater (has copies of gradient collectors)
update_worker = UWorker(
    col_fraction_workers=1.0,
    grad_execution=prl.CENTRAL,  # TODO: remove! this can be known from len(gradient_workers)
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
