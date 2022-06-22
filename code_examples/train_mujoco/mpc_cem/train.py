#!/usr/bin/env python3

import os
import sys
import time
import wandb
import argparse

from pytorchrl.learner import Learner
from pytorchrl.scheme import Scheme
from pytorchrl.agent.env import VecEnv
from pytorchrl.agent.algorithms import MPC_CEM
from pytorchrl.agent.storages import MBReplayBuffer
from pytorchrl.agent.actors.world_models import WorldModel
from pytorchrl.agent.actors import ModelBasedPlannerActor
from pytorchrl.envs.mujoco import mujoco_train_env_factory, mujoco_test_env_factory
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

        # Sanity check, make sure that logging matches execution
        args = wandb.config

        # 1. Define Train Vector of Envs
        train_envs_factory, action_space, obs_space = VecEnv.create_factory(
            vec_env_size=args.num_env_processes,
            log_dir=args.log_dir,
            env_fn=mujoco_train_env_factory,
            env_kwargs={"env_id": args.env_id,
                        "frame_skip": args.frame_skip,
                        "frame_stack": args.frame_stack})

        # 2. Define Test Vector of Envs (Optional)
        test_envs_factory, _, _ = VecEnv.create_factory(
            vec_env_size=args.num_env_processes,
            log_dir=args.log_dir,
            env_fn=mujoco_test_env_factory,
            env_kwargs={"env_id": args.env_id,
                        "frame_skip": args.frame_skip,
                        "frame_stack": args.frame_stack})

        # 3. Define RL training algorithm
        algo_factory, algo_name = MPC_CEM.create_factory(
            lr=args.lr, start_steps=args.start_steps,
            update_every=args.update_every, mb_epochs=args.mb_epochs,
            action_noise=args.action_noise, mini_batch_size=args.mini_batch_size,
            test_every=args.test_every)

        # 4. Define Actor
        actor_factory = ModelBasedPlannerActor.create_factory(
            obs_space, action_space, algo_name, horizon=args.horizon,
            n_planner=args.n_planner, restart_model=args.restart_model, world_model_class=WorldModel,
            world_model_kwargs={"hidden_size": args.hidden_size, "reward_function": None})

        # 5. Define rollouts storage
        storage_factory = MBReplayBuffer.create_factory(size=args.buffer_size)

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

        scheme = Scheme(**params)

        # 7. Define learner
        learner = Learner(scheme, target_steps=args.num_env_steps, log_dir=args.log_dir)

        # 8. Define train loop
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
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile)

    # Wandb
    parser.add_argument(
        '--experiment-name', default=None,
        help='Name of the wandb experiment the agent belongs to')
    parser.add_argument(
        '--agent-name', default=None, help='Name of the wandb run')
    parser.add_argument(
        '--wandb-key', default=None, help='Init key from wandb account')

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

    # MPC specs
    parser.add_argument(
        '--learn-reward-function', default=False, action='store_true',
        help='Learns the reward function if set, (default: False)')
    parser.add_argument(
        '--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument(
        '--mb-epochs', type=int, default=60,
        help='Number of epochs to train the dynamics model, (default: 60)')
    parser.add_argument(
        '--eps', type=float, default=1e-8,
        help='Adam optimizer epsilon (default: 1e-8)')
    parser.add_argument(
        '--n-planner', type=int, default=500,
        help='Number of parallel planner for each actor (default: 500)')
    parser.add_argument(
        '--horizon', type=int, default=12,
        help='The horizon of online planning (default: 12)')
    parser.add_argument(
        '--mpc-type', type=str, choices=["RS", "CEM", "PDDM"], default="RS",
        help='Type of MPC optimizer, RS: Random Shooting, CEM: Cross Entropy Method (default: RS)')
    parser.add_argument(
        '--action-noise', default=False, action='store_true',
        help='Adding noise to the actions, (default: False)')
    parser.add_argument(
        '--start-steps', type=int, default=5000,
        help='Number of initial random steps (default: 5000)')
    parser.add_argument(
        '--buffer-size', type=int, default=10000,
        help='Rollouts storage size (default: 10000 transitions)')
    parser.add_argument(
        '--update-every', type=int, default=50,
        help='Num env collected steps between SAC network update stages (default: 50)')
    parser.add_argument(
        '--num-updates', type=int, default=50,
        help='Num network updates per dynamics network update stage (default 50)')
    parser.add_argument(
        '--mini-batch-size', type=int, default=32,
        help='Mini batch size for network updates (default: 32)')
    parser.add_argument("--test-every", type=int, default=1, help="")

    # CEM parameter
    parser.add_argument(
        '--iter-update-steps', type=int, default=3,
        help='Iterative update steps for CEM (default: 3)')
    parser.add_argument(
        '--k-best', type=int, default=5,
        help='K-Best members of the mean prediction forming the next mean distribution')
    parser.add_argument(
        '--update-alpha', type=float, default=0.0,
        help='Soft update alpha for each iteration (default: 0.0)')

    # PDDM parameter
    parser.add_argument(
        '--gamma', type=float, default=1.0,
        help='PDDM gamma value (default: 1.0)')
    parser.add_argument(
        "--beta", type=float, default=0.5,
        help='PDDM beta value (default: 0.5)')

    # Feature dynamics model specs
    parser.add_argument(
        '--hidden-size', type=int, default=500,
        help='Number of hidden nodes for the dynamics model (default: 500)')
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
