#!/usr/bin/env python3

import os
import sys
import time
import json
import wandb
import argparse
import numpy as np

import torch
from pytorchrl.learner import Learner
from pytorchrl.scheme import Scheme
from pytorchrl.agent.algorithms import PPO
from pytorchrl.agent.env import VecEnv
from pytorchrl.agent.storages import GAEBuffer
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor
from pytorchrl.envs.atari import atari_train_env_factory, atari_test_env_factory
from pytorchrl.utils import LoadFromFile, save_argparse, cleanup_log_dir


def main():
    args = get_args()
    cleanup_log_dir(args.log_dir)
    save_argparse(args, os.path.join(args.log_dir, "conf.yaml"), [])

    # Handle wandb init
    if args.wandb_key:
        mode = "online"
        wandb.login(key=str(args.wandb_key))
    else:
        mode = "disabled"

    with wandb.init(project=args.experiment_name, name=args.agent_name, config=args, mode=mode):

        info_keywords = []
        if args.episodic_life:
            info_keywords += ['EpisodicReward', 'Lives']
        if args.clip_rewards:
            info_keywords += ['UnclippedReward']

        # Component 1: Train Vector of Envs
        train_envs_factory, action_space, obs_space = VecEnv.create_factory(
            env_fn=atari_train_env_factory,
            env_kwargs={
                "env_id": args.env_id,
                "frame_stack": args.frame_stack,
            },
            vec_env_size=args.num_env_processes, log_dir=args.log_dir,
            info_keywords=tuple(info_keywords))

        # Component 2:  Test Vector of Envs (Optional)
        test_envs_factory, _, _ = VecEnv.create_factory(
            env_fn=atari_test_env_factory,
            env_kwargs={"env_id": args.env_id, "frame_stack": args.frame_stack},
            vec_env_size=args.num_env_processes, log_dir=args.log_dir)

        # Component 3: RL algorithm
        algo_factory, algo_name = PPO.create_factory(
            lr=args.lr, num_epochs=args.ppo_epoch, clip_param=args.clip_param,
            entropy_coef=args.entropy_coef, value_loss_coef=args.value_loss_coef,
            max_grad_norm=args.max_grad_norm, num_mini_batch=args.num_mini_batch,
            use_clipped_value_loss=args.use_clipped_value_loss, gamma=args.gamma)

        # Component 4: RL Actor
        actor_factory = OnPolicyActor.create_factory(
            obs_space, action_space, algo_name, restart_model=args.restart_model)

        # Component 5: Data Storage
        storage_factory = GAEBuffer.create_factory(size=args.num_steps, gae_lambda=args.gae_lambda)

        # Component 6: Data collector (has copies of agent modules: envs, actor, storage, algo)
        col_workers_factory = CWorkerSet.create_factory(
            num_workers=1,
            algo_factory=algo_factory,
            actor_factory=actor_factory,
            storage_factory=storage_factory,
            test_envs_factory=test_envs_factory,
            train_envs_factory=train_envs_factory,
            col_worker_resources={"num_cpus": 1, "num_gpus": 0.25},
        )
        # Component 7: Gradient Collector (has copies of data collectors)
        grad_workers_factory = GWorkerSet.create_factory(
            num_workers=1,
            col_communication=prl.SYNC,
            col_workers_factory=col_workers_factory,
            grad_worker_resources={"num_cpus": 1, "num_gpus": 0.25},
        )
        # Component 8: Model Updater (has copies of gradient collectors)
        update_worker = UWorker(
            grad_communication=prl.SYNC,
            grad_workers_factory=grad_workers_factory,
        )
        collected_steps = 0
        while collected_steps < target_steps:
            # Collect data, take one grad step, update model parameters
            print("step!")
            info = update_worker.step()

        # start_time = time.time()
        # collected_steps = 0
        # target_steps = args.num_env_steps
        # total_updates = np.ceil((args.num_env_steps * args.ppo_epoch * args.num_mini_batch) / (
        #         args.num_env_processes * args.num_steps)) + args.ppo_epoch * args.num_mini_batch
        # alpha = np.linspace(1.0, 0.0, int(total_updates))
        #
        # while collected_steps < target_steps:
        #
        #     # Collect data, take one grad step, update model parameters
        #     info = self.update_worker.step()
        #
        #     # Update counter
        #     self.num_samples_collected += info.pop(prl.NUMSAMPLES)
        #
        #     # Update learning rate and clip param
        #     update_worker.update_algorithm_parameter("lr", alpha[updates] * args.lr)
        #     update_worker.update_algorithm_parameter("clip_param", alpha[updates] * args.clip_param)
        #
        #     if args.max_time != -1 and (time.time() - start_time) > args.max_time:
        #         break

        print("Finished!")
        sys.exit()


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # Configuration file, keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile)

    # Wandb
    parser.add_argument(
        '--experiment_name', default=None, help='Name of the wandb experiment the agent belongs to')
    parser.add_argument(
        '--agent_name', default=None, help='Name of the wandb run')
    parser.add_argument(
        '--wandb_key', default=None, help='Init key from wandb account')

    # Environment specs
    parser.add_argument(
        '--env-id', type=str, default=None, help='Gym environment id (default None)')
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action (default no skip)')
    parser.add_argument(
        '--frame-stack', type=int, default=0,
        help='Number of frame to stack in observation (default no stack)')
    parser.add_argument(
        '--clip_rewards', action='store_true', default=False,
        help='Clip environment rewards')
    parser.add_argument(
        '--episodic_life', action='store_true', default=False,
        help='Turn every life into an episode')

    # PPO specs
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--gae-lambda', type=float, default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef', type=float, default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef', type=float, default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--use_clipped_value_loss', action='store_true', default=False,
        help='clip value loss update')
    parser.add_argument(
        '--num-steps', type=int, default=20000,
        help='number of forward steps in PPO (default: 20000)')
    parser.add_argument(
        '--ppo-epoch', type=int, default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch', type=int, default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param', type=float, default=0.2,
        help='ppo clip parameter (default: 0.2)')

    # Feature extractor model specs
    parser.add_argument(
        '--feature-extractor-net', default='MLP', help='Type of nn. Options include MLP, CNN, Fixup')
    parser.add_argument(
        '--restart-model', default=None,
        help='Restart training using the model given')
    parser.add_argument(
        '--recurrent-net', default=None, help='Recurrent neural networks to use')

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
        '--log-dir', default='/tmp/atari_ppo',
        help='directory to save agent logs (default: /tmp/atari_ppo)')

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args


if __name__ == "__main__":
    main()
