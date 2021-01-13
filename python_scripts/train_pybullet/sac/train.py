import os
import ray
import sys
import time
import json
import argparse

from pytorchrl import utils
from pytorchrl import Learner
from pytorchrl.scheme import Scheme
from pytorchrl.agent.algos import SAC
from pytorchrl.agent.env import VecEnv
from pytorchrl.agent.storages import ReplayBuffer
from pytorchrl.agent.actors import OffPolicyActor, get_feature_extractor
from pytorchrl.envs import pybullet_train_env_factory, pybullet_test_env_factory


def main():

    args = get_args()
    utils.cleanup_log_dir(args.log_dir)
    args_dict = vars(args)
    json.dump(args_dict, open(os.path.join(args.log_dir, "training_arguments.json"), "w"), indent=4)

    if args.cluster:
        ray.init(address="auto")
    else:
        ray.init()

    resources = ""
    for k, v in ray.cluster_resources().items():
        resources += "{} {}, ".format(k, v)
    print(resources[:-2], flush=True)

    # 1. Define Train Vector of Envs
    train_envs_factory, action_space, obs_space = VecEnv.create_factory(
        vec_env_size=args.num_env_processes, log_dir=args.log_dir,
        env_fn=pybullet_train_env_factory, env_kwargs={
            "env_id": args.env_id,
            "frame_skip": args.frame_skip,
            "frame_stack": args.frame_stack})

    # 2. Define Test Vector of Envs (Optional)
    test_envs_factory, _, _ = VecEnv.create_factory(
        vec_env_size=args.num_env_processes, log_dir=args.log_dir,
        env_fn=pybullet_test_env_factory, env_kwargs={
            "env_id": args.env_id,
            "frame_skip": args.frame_skip,
            "frame_stack": args.frame_stack})

    # 3. Define RL training algorithm
    algo_factory = SAC.create_factory(
        lr_pi=args.lr, lr_q=args.lr, lr_alpha=args.lr, initial_alpha=args.alpha,
        gamma=args.gamma, polyak=args.polyak, num_updates=args.num_updates,
        update_every=args.update_every, start_steps=args.start_steps,
        mini_batch_size=args.mini_batch_size)

    # 4. Define RL Policy
    actor_factory = OffPolicyActor.create_factory(
        obs_space, action_space,
        feature_extractor_network=get_feature_extractor(args.nn),
        recurrent_policy=args.recurrent_policy,
        restart_model=args.restart_model)

    # 5. Define rollouts storage
    storage_factory = ReplayBuffer.create_factory(size=args.buffer_size)

    # 6. Define scheme
    params = {}

    # add core modules
    params.update({
        "algo_factory": algo_factory,
        "actor_factory": actor_factory,
        "storage_factory": storage_factory,
        "train_envs_factory": train_envs_factory,
        "test_envs_factory": test_envs_factory,
    })

    # add collection specs
    params.update({
        "num_col_workers": args.num_col_workers,
        "col_communication": args.com_col_workers,
        "col_worker_resources": {"num_cpus": 1, "num_gpus": 0.5},
        "sync_col_specs": {"fraction_samples": 1.0, "fraction_workers": 1.0}
    })

    # add gradient specs
    params.update({
        "num_grad_workers": args.num_grad_workers,
        "grad_communication": args.com_grad_workers,
        "grad_worker_resources": {"num_cpus": 1.0, "num_gpus": 0.5},
    })

    scheme = Scheme(**params)

    # 7. Define learner
    learner = Learner(scheme, target_steps=args.num_env_steps, log_dir=args.log_dir)

    # 8. Define train loop
    iterations = 0
    start_time = time.time()
    while not learner.done():

        learner.step()

        if iterations % args.log_interval == 0:
            learner.print_info()

        if iterations % args.save_interval == 0:
            save_name = learner.save_model()
            args_dict.update({"latest_model": save_name})
            args_path = os.path.join(args.log_dir, "training_arguments.json")
            json.dump(args_dict, open(args_path, "w"), indent=4)

        if args.max_time != -1 and (time.time() - start_time) > args.max_time:
            break

        iterations += 1

    print("Finished!")
    sys.exit()

def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # Environment specs
    parser.add_argument(
        '--env-id', type=str, default=None,
        help='Gym environment id (default None)')
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action (default no skip)')
    parser.add_argument(
        '--frame-stack', type=int, default=0,
        help='Number of frame to stack in observation (default no stack)')

    # SAC specs
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps', type=float, default=1e-8,
        help='Adam optimizer epsilon (default: 1e-8)')
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--alpha', type=float, default=0.2,
        help='SAC alpha parameter (default: 0.2)')
    parser.add_argument(
        '--polyak', type=float, default=0.995,
        help='SAC polyak paramater (default: 0.995)')
    parser.add_argument(
        '--start-steps', type=int, default=1000,
        help='SAC num initial random steps (default: 1000)')
    parser.add_argument(
        '--buffer-size', type=int, default=10000,
        help='Rollouts storage size (default: 10000 transitions)')
    parser.add_argument(
        '--update-every', type=int, default=50,
        help='Num env collected steps between SAC network update stages (default: 50)')
    parser.add_argument(
        '--num-updates', type=int, default=50,
        help='Num network updates per SAC network update stage (default 50)')
    parser.add_argument(
        '--mini-batch-size', type=int, default=32,
        help='Mini batch size for network updates (default: 32)')
    parser.add_argument(
        '--target-update-interval', type=int, default=1,
        help='Num SAC network updates per target network updates (default: 1)')

    # Feature extractor model specs
    parser.add_argument(
        '--nn', default='MLP', help='Type of nn. Options are MLP, CNN, Fixup')
    parser.add_argument(
        '--restart-model', default=None,
        help='Restart training using the model given')
    parser.add_argument(
        '--recurrent-policy', action='store_true', default=False,
        help='Use a recurrent policy')

    # Scheme specs
    parser.add_argument(
        '--num-env-processes', type=int, default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-grad-workers', type=int, default=0,
        help='how many agent workers to use (default: 1)')
    parser.add_argument(
        '--com-grad-workers', default='synchronised',
        help='communication patters grad workers (default: synchronised)')
    parser.add_argument(
        '--num-col-workers', type=int, default=0,
        help='how many agent workers to use (default: 1)')
    parser.add_argument(
        '--com-col-workers', default='synchronised',
        help='communication patters col workers (default: synchronised)')
    parser.add_argument(
        '--cluster', action='store_true', default=False,
        help='script is running in a cluster')

    # General training specs
    parser.add_argument(
        '--num-env-steps', type=int, default=10e7,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--max-time', type=int, default=-1,
        help='stop script after this amount of time in seconds (default: no limit)')
    parser.add_argument(
        '--log-interval', type=int, default=1,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval', type=int, default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--log-dir', default='/tmp/ppo/',
        help='directory to save agent logs (default: /tmp/ppo)')

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args


if __name__ == "__main__":
    main()
