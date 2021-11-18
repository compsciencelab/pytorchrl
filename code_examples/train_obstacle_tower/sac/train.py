#!/usr/bin/env python3

import os
import ray
import sys
import time
import wandb
import argparse

from pytorchrl.learner import Learner
from pytorchrl.scheme import Scheme
from pytorchrl.agent.algorithms import SAC
from pytorchrl.agent.env import VecEnv
from pytorchrl.agent.storages import ReplayBuffer, NStepReplayBuffer, PERBuffer, EREBuffer
from pytorchrl.agent.actors import OffPolicyActor, get_feature_extractor
from pytorchrl.envs.obstacle_tower.obstacle_tower_env_factory import obstacle_train_env_factory
from pytorchrl.utils import LoadFromFile, save_argparse, cleanup_log_dir


def main():

    args = get_args()
    cleanup_log_dir(args.log_dir)
    save_argparse(args, os.path.join(args.log_dir, "conf.yaml"), [])

    if args.cluster:
        ray.init(address="auto")
    else:
        ray.init(num_cpus=0)

    resources = ""
    for k, v in ray.cluster_resources().items():
        resources += "{} {}, ".format(k, v)
    print(resources[:-2], flush=True)

    # Handle wandb init
    if args.wandb_key:
        mode = "online"
        wandb.login(key=str(args.wandb_key))
    else:
        mode = "disabled"

    with wandb.init(project=args.experiment_name, name=args.agent_name, config=args, mode=mode):

        # Sanity check, make sure that logging matches execution
        args = wandb.config

        # Define Train Vector of Envs
        train_envs_factory, action_space, obs_space = VecEnv.create_factory(
            env_fn=obstacle_train_env_factory,
            env_kwargs={
                "frame_skip": args.frame_skip,
                "frame_stack": args.frame_stack,
                "reward_shape": args.reward_shape,
                "reduced_actions": args.reduced_action_space,
                "num_actions": args.num_actions,
            },
            vec_env_size=args.num_env_processes, log_dir=args.log_dir,
            info_keywords=('floor', 'start', 'seed'))

        # Define RL training algorithm
        algo_factory, algo_name = SAC.create_factory(
            lr_pi=args.lr, lr_q=args.lr, lr_alpha=args.lr, initial_alpha=args.alpha,
            gamma=args.gamma, polyak=args.polyak, num_updates=args.num_updates,
            update_every=args.update_every, start_steps=args.start_steps,
            mini_batch_size=args.mini_batch_size)

        # Define RL Policy
        actor_factory = OffPolicyActor.create_factory(
            obs_space, action_space, algo_name, recurrent_nets=args.recurrent_nets,
            restart_model=args.restart_model, obs_feature_extractor=args.nn)

        # 5. Define rollouts storage
        storage_factory = ReplayBuffer.create_factory(size=args.buffer_size)
        # storage_factory = NStepReplayBuffer.create_factory(size=args.buffer_size, n_step=2)
        # storage_factory = PERBuffer.create_factory(size=args.buffer_size, epsilon=0.0, alpha=0.6, beta=0.6)
        # storage_factory = EREBuffer.create_factory(size=args.buffer_size, eta=0.996, cmin=5000)

        # Define scheme
        params = {}

        # Add core modules
        params.update({
            "algo_factory": algo_factory,
            "actor_factory": actor_factory,
            "storage_factory": storage_factory,
            "train_envs_factory": train_envs_factory,
        })

        # Add collection specs
        params.update({
            "num_col_workers": args.num_col_workers,
            "col_workers_communication": args.com_col_workers,
            "col_workers_resources": {"num_cpus": 1.0, "num_gpus": 0.5},
        })

        # Add gradient specs
        params.update({
            "num_grad_workers": args.num_grad_workers,
            "grad_workers_communication": args.com_grad_workers,
            "grad_workers_resources": {"num_cpus": 1.0, "num_gpus": 0.5},
        })

        scheme = Scheme(**params)

        # Define learner
        learner = Learner(scheme, target_steps=args.num_env_steps, log_dir=args.log_dir)

        # Define train loop
        iterations = 0
        start_time = time.time()
        while not learner.done():

            learner.step()

            if iterations % args.log_interval == 0:
                log_data = learner.get_metrics(add_episodes_metrics=True)
                log_data = {k.split("/")[-1]: v for k, v in log_data.items()}
                wandb.log(log_data, step=learner.num_samples_collected)
                learner.print_info()

            if iterations % args.save_interval == 0:
                save_name = learner.save_model()

            if args.max_time != -1 and (time.time() - start_time) > args.max_time:
                break

            iterations += 1

        print("Finished!")
        sys.exit()


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # Configuration file, keep first
    parser.add_argument('--conf','-c', type=open, action=LoadFromFile)

    # Wandb
    parser.add_argument(
        '--experiment_name', default=None, help='Name of the wandb experiment the agent belongs to')
    parser.add_argument(
        '--agent_name', default=None, help='Name of the wandb run')
    parser.add_argument(
        '--wandb_key', default=None, help='Init key from wandb account')

    # Environment specs
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action (default no skip)')
    parser.add_argument(
        '--frame-stack', type=int, default=0,
        help='Number of frame to stack in observation (default no stack)')
    parser.add_argument(
        '--reward-shape', action='store_true', default=False,
        help='Whether or not to use reward shaping')
    parser.add_argument(
        '--reduced-action-space', action='store_true', default=False,
        help='Whether or not to use a reduced action space.')
    parser.add_argument(
        '--num-actions', type=int, default=6,
        help='Size of the reduced action space (6, 7 or 8) (default: 6)')

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
        '--recurrent-nets', action='store_true', default=False,
        help='Use a recurrent policy')

    # Scheme specs
    parser.add_argument(
        '--num-env-processes', type=int, default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-grad-workers', type=int, default=1,
        help='how many agent workers to use (default: 1)')
    parser.add_argument(
        '--com-grad-workers', default='synchronous',
        help='communication patters grad workers (default: synchronous)')
    parser.add_argument(
        '--num-col-workers', type=int, default=1,
        help='how many agent workers to use (default: 1)')
    parser.add_argument(
        '--com-col-workers', default='synchronous',
        help='communication patters col workers (default: synchronous)')
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
        '--log-dir', default='/tmp/pybullet_sac',
        help='directory to save agent logs (default: /tmp/pybullet_sac)')

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args


if __name__ == "__main__":
    main()
