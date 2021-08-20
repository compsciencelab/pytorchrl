#!/usr/bin/env python3

import os
import ray
import gym
import sys
import time
import argparse

from pytorchrl.learner import Learner
from pytorchrl.scheme import Scheme
from pytorchrl.agent.algorithms import PPO
from pytorchrl.agent.env import VecEnv
from pytorchrl.agent.storages import GAEBuffer
from pytorchrl.envs.common import FrameStack, FrameSkip
from pytorchrl.envs import pybullet_train_env_factory, pybullet_test_env_factory
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor
from pytorchrl.utils import LoadFromFile, save_argparse, cleanup_log_dir

# Testing
import ipdb; ipdb.set_trace()
from pytorchrl.agent.storages.on_policy.ppod_buffer import PPODBuffer

gym.envs.register(
    id='SparseReacher-v1',
    entry_point='pytorchrl.envs.pybullet.sparse_reacher:SparseReacherBulletEnv',
	max_episode_steps=150,
	reward_threshold=18.0,
)

def sparse_reacher_env_factory(seed=0, index_worker=0, index_env=0, frame_skip=0, frame_stack=1):
    """ Create sparse sparse_reacher environment instances."""

    env = gym.make("SparseReacher-v1")
    env.seed(seed + index_worker + index_env)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    return env

def main():

    args = get_args()
    cleanup_log_dir(args.log_dir)
    save_argparse(args, os.path.join(args.log_dir, "conf.yaml"),[])

    if args.cluster:
        ray.init(address="auto")
    else:
        ray.init(num_cpus=0)

    resources = ""
    for k, v in ray.cluster_resources().items():
        resources += "{} {}, ".format(k, v)
    print(resources[:-2], flush=True)

    # # 1. Define Train Vector of Envs
    # train_envs_factory, action_space, obs_space = VecEnv.create_factory(
    #     vec_env_size=args.num_env_processes, log_dir=args.log_dir,
    #     env_fn=sparse_reacher_env_factory, env_kwargs={
    #         # "seed": args.seed,
    #         "frame_skip": args.frame_skip,
    #         "frame_stack": args.frame_stack})

    # 1. Define Train Vector of Envs
    train_envs_factory, action_space, obs_space = VecEnv.create_factory(
        vec_env_size=args.num_env_processes, log_dir=args.log_dir,
        env_fn=pybullet_train_env_factory, env_kwargs={
            "env_id": args.env_id,
            "reward_delay": 1,
            "frame_skip": args.frame_skip,
            "frame_stack": args.frame_stack})

    # 2. Define RL training algorithm
    algo_factory, algo_name = PPO.create_factory(
        lr=args.lr, num_epochs=args.ppo_epoch, clip_param=args.clip_param,
        entropy_coef=args.entropy_coef, value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm, num_mini_batch=args.num_mini_batch,
        gamma=args.gamma)

    # 3. Define RL Policy
    actor_factory = OnPolicyActor.create_factory(
        obs_space, action_space, algo_name, restart_model=args.restart_model)

    # 4. Define rollouts storage
    storage_factory = PPODBuffer.create_factory(
        size=args.num_steps,
        initial_demos_dir="/tmp/demos",
        target_demos_dir="/tmp/demos",
        gae_lambda=args.gae_lambda,
    )

    # 5. Define scheme
    params = {}

    # add core modules
    params.update({
        "algo_factory": algo_factory,
        "actor_factory": actor_factory,
        "storage_factory": storage_factory,
        "train_envs_factory": train_envs_factory,
    })

    # add collection specs
    params.update({
        "num_col_workers": args.num_col_workers,
        "col_workers_communication": args.com_col_workers,
        "col_workers_resources": {"num_cpus": 1, "num_gpus": 0.5},
    })

    # add gradient specs
    params.update({
        "num_grad_workers": args.num_grad_workers,
        "grad_workers_communication": args.com_grad_workers,
        "grad_workers_resources": {"num_cpus": 1.0, "num_gpus": 0.5},
    })

    scheme = Scheme(**params)

    # 6. Define learner
    learner = Learner(scheme, target_steps=args.num_env_steps, log_dir=args.log_dir)

    # 7. Define train loop
    iterations = 0
    start_time = time.time()
    while not learner.done():

        learner.step()

        if iterations % args.log_interval == 0:
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
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile)

    # Environment specs
    parser.add_argument(
        '--env-id', type=str, default=None,
        help='Gym environment id (default None)')
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action (default no skip)')
    parser.add_argument(
        '--frame-stack', type=int, default=1,
        help='Number of frame to stack in observation (default no stack)')

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
        '--num-grad-workers', type=int, default=1,
        help='how many agent workers to use (default: 1)')
    parser.add_argument(
        '--com-grad-workers', default='synchronised',
        help='communication patters grad workers (default: synchronised)')
    parser.add_argument(
        '--num-col-workers', type=int, default=1,
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
        '--log-dir', default='/tmp/pybullet_ppo',
        help='directory to save agent logs (default: /tmp/pybullet_ppo)')

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args


if __name__ == "__main__":
    main()
