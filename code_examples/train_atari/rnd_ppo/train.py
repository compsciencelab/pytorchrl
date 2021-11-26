#!/usr/bin/env python3
import sys
import subprocess


# Install required pip packages
def install(package): subprocess.check_call([sys.executable, "-m", "pip", "install", package])


import os
import ray
import time
import glob
import yaml
import torch
import wandb
import shutil
import random
import argparse
import numpy as np
import torch.nn as nn

from pytorchrl.learner import Learner
from pytorchrl.scheme import Scheme
from pytorchrl.agent.algorithms import RND_PPO
from pytorchrl.agent.env import VecEnv
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor
from pytorchrl.envs.atari import atari_train_env_factory
from pytorchrl.utils import LoadFromFile, save_argparse, cleanup_log_dir
from pytorchrl.agent.storages import GAEBuffer

def main():

    print("GPU available: {}".format(torch.cuda.is_available()))

    # Get and log config
    args = get_args()

    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Handle Ray init
    if args.cluster:
        ray.init(address="auto")
    else:
        ray.init(
            num_cpus=0,
            object_store_memory=1024 ** 3 * 3,
            _redis_max_memory=1024 ** 3 * 1,
            _memory=1024 ** 3 * 1,
            _driver_object_store_memory=1024 ** 3 * 1)

    # Handle wandb init
    if args.wandb_key:
        mode = "online"
        wandb.login(key=str(args.wandb_key))
    else:
        mode = "disabled"

    with wandb.init(project=args.experiment_name, name=args.agent_name, config=args, mode=mode):

        # Sanity check, make sure that logging matches execution
        args = wandb.config

        info_keywords = ["VisitedRooms"]
        if args.episodic_life:
            info_keywords += ['EpisodicReward', 'Lives']
        if args.clip_rewards:
            info_keywords += ['ClippedReward']

        # Define Train Vector of Envs
        train_envs_factory, action_space, obs_space = VecEnv.create_factory(
            env_fn=atari_train_env_factory,
            env_kwargs={
                "env_id": args.env_id,
                "frame_stack": args.frame_stack,
                "episodic_life": args.episodic_life,
                "clip_rewards": args.clip_rewards,
            },
            vec_env_size=args.num_env_processes, log_dir=args.log_dir,
            info_keywords=tuple(info_keywords))

        # Define RL training algorithm
        algo_factory, algo_name = RND_PPO.create_factory(
            lr=args.lr, num_epochs=args.ppo_epoch, clip_param=args.clip_param,
            entropy_coef=args.entropy_coef, value_loss_coef=args.value_loss_coef,
            max_grad_norm=args.max_grad_norm, num_mini_batch=args.num_mini_batch,
            use_clipped_value_loss=False, gamma_intrinsic=args.gamma_intrinsic,
            ext_adv_coeff=args.ext_adv_coeff, int_adv_coeff=args.int_adv_coeff,
            predictor_proportion=args.predictor_proportion, gamma=args.gamma,
            pre_normalization_length=args.pre_normalization_length,
            pre_normalization_steps=args.pre_normalization_steps)

        # Look for available model checkpoint in log_dir - node failure case
        checkpoints = sorted(glob.glob(os.path.join(args.log_dir, "model.state_dict*")))
        if len(checkpoints) > 0:
            checkpoint = checkpoints[-1]
            print("Loading model from {}\n".format(checkpoint))
        else:
            print("Training model from scratch\n")
            checkpoint = None

        # Define RL Policy
        actor_factory = OnPolicyActor.create_factory(
            obs_space, action_space, algo_name,
            feature_extractor_network=get_feature_extractor("CNN"),
            restart_model=checkpoint, recurrent_nets=False)

        # Define rollouts storage
        storage_factory = GAEBuffer.create_factory(size=args.num_steps, gae_lambda=args.gae_lambda)

        # Define scheme
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
            "col_workers_resources": {"num_cpus": 1.0, "num_gpus": 0.5},
        })

        # add gradient specs
        params.update({
            "num_grad_workers": args.num_grad_workers,
            "grad_workers_communication": args.com_grad_workers,
            "grad_workers_resources": {"num_cpus": 1.0, "num_gpus": 0.5},
        })

        scheme = Scheme(**params)
        wandb.config.update(scheme.get_agent_components())  # Log agent components

        # Define learner
        training_steps = args.target_env_steps - args.start_env_steps
        learner = Learner(scheme, target_steps=training_steps, log_dir=args.log_dir)

        # Define train loop
        iterations = 0
        save_name = None
        previous_save_name = None
        start_time = time.time()
        while not learner.done():

            learner.step()

            if iterations % args.log_interval == 0:
                log_data = learner.get_metrics(add_episodes_metrics=True)
                log_data = {k.split("/")[-1]: v for k, v in log_data.items()}
                wandb.log(log_data, step=learner.num_samples_collected + args.start_env_steps)
                learner.print_info()

                with open(os.path.join(args.log_dir, "progress.txt"), "w+") as progressfile:
                    progressfile.write("{:.2f}".format(learner.num_samples_collected / training_steps))

            if iterations % args.save_interval == 0:
                # Save current model version
                previous_save_name = save_name
                save_name = learner.save_model()
                # Remove previous model version
                if previous_save_name:
                    os.remove(previous_save_name)

            if args.max_time != -1 and (time.time() - start_time) > args.max_time:
                break

            iterations += 1

    # Save latest model version
    previous_save_name = save_name
    save_name = learner.save_model()

    # Remove previous model version
    os.remove(previous_save_name)
    os.remove(save_name)

    # Define results dir
    shutil.move(
        os.path.join(args.log_dir, "monitor_logs"),
        os.path.join(args.log_dir, "results")
    )

    # Assert model.state_dict exists, and move it to results
    assert os.path.exists(os.path.join(os.getcwd(), "model.state_dict"))
    shutil.move(
        os.path.join(os.getcwd(), "model.state_dict"),
        os.path.join(os.getcwd(), "results/model.state_dict")
    )

    # Zip results
    logs_path = os.path.join(args.log_dir, "results")
    shutil.make_archive(logs_path, "zip", logs_path)

    with open(os.path.join(args.log_dir, "progress.txt"), "w+") as progressfile:
        progressfile.write("{:.2f}".format(1.00))

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
        '--env-id', type=str, default=None,
        help='Gym environment id (default None)')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Env seed (default 0)')
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action (default no skip)')
    parser.add_argument(
        '--frame-stack', type=int, default=1,
        help='Number of frame to stack in observation (default no stack)')
    parser.add_argument(
        '--clip-rewards', action='store_true', default=False,
        help='Clip env rewards between -1 and 1')
    parser.add_argument(
        '--episodic-life', action='store_true', default=False,
        help='Treat end-of-life as end-of-episode')

    # RND PPO specs
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
    parser.add_argument(
        '--gamma-intrinsic', type=float, default=0.99,
        help='rnd ppo intrinsic gamma parameter (default: 0.99)')
    parser.add_argument(
        '--ext-adv-coeff', type=float, default=2.0,
        help='rnd ppo external advantage coefficient parameter (default: 2.0)')
    parser.add_argument(
        '--int-adv-coeff', type=float, default=1.0,
        help='rnd ppo internal advantage coefficient parameter (default: 1.0)')
    parser.add_argument(
        '--predictor-proportion', type=float, default=1.0,
        help='rnd ppo proportion of batch samples to use to update predictor net (default: 1.0)')
    parser.add_argument(
        '--pre-normalization-steps', type=int, default=50,
        help='rnd ppo number of pre-normalization steps parameter (default: 50)')

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
        '--start-env-steps', type=int, default=0,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--target-env-steps', type=int, default=10e7,
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

    return args


if __name__ == "__main__":
    main()