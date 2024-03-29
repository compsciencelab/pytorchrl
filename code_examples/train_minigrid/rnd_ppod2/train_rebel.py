#!/usr/bin/env python3

import os
import sys
import time
import json
import wandb
import argparse
import numpy as np
import torch.nn as nn

from pytorchrl.learner import Learner
from pytorchrl.scheme import Scheme
from pytorchrl.agent.algorithms import RND_PPO
from pytorchrl.agent.env import VecEnv
from pytorchrl.agent.storages import PPOD2RebelBuffer
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor
from pytorchrl.envs.minigrid.minigrid_env_factory import minigrid_train_env_factory
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
            env_fn=minigrid_train_env_factory,
            env_kwargs={"env_id": args.env_id},
            vec_env_size=args.num_env_processes, log_dir=args.log_dir)

        # 2. Define RL training algorithm
        algo_factory, algo_name = RND_PPO.create_factory(
            lr=args.lr, num_epochs=args.ppo_epoch, clip_param=args.clip_param,
            entropy_coef=args.entropy_coef, value_loss_coef=args.value_loss_coef,
            max_grad_norm=args.max_grad_norm, num_mini_batch=args.num_mini_batch,
            use_clipped_value_loss=False, gamma_intrinsic=args.gamma_intrinsic,
            ext_adv_coeff=args.ext_adv_coeff, int_adv_coeff=args.int_adv_coeff,
            predictor_proportion=args.predictor_proportion, gamma=args.gamma,
            pre_normalization_steps=args.pre_normalization_steps,
            pre_normalization_length=args.num_steps,
            intrinsic_rewards_network=get_feature_extractor(args.feature_extractor_net),
            intrinsic_rewards_target_network_kwargs={
                "output_sizes": [512],
                 "activation": nn.LeakyReLU,
                "final_activation": False,
                "rgb_norm": False,
            },
            intrinsic_rewards_predictor_network_kwargs={
                "output_sizes": [512, 512, 512],
                 "activation": nn.LeakyReLU,
                "final_activation": False,
                "rgb_norm": False,
            },
        )

        # 3. Define rollouts storage
        storage_factory = PPOD2RebelBuffer.create_factory(
            size=args.num_steps, gae_lambda=args.gae_lambda,
            reward_predictor_factory=get_feature_extractor(args.feature_extractor_net),
            reward_predictor_net_kwargs={
                "input_space": obs_space,
                "output_sizes": [256, 448, 1],
                "final_activation": False,
            },
            restart_reward_predictor_net=args.restart_reference_model,
            target_reward_demos_dir=os.path.join(args.log_dir, "reward_demos"),
            initial_reward_threshold=args.initial_reward_threshold)

        # 4. Define RL Policy
        actor_factory = OnPolicyActor.create_factory(
            obs_space, action_space, algo_name,
            restart_model={
                "policy_net": args.restart_model,
                "value_net": args.restart_model,
            },
            shared_policy_value_network=args.shared_policy_value_network,
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

        scheme = Scheme(**params)

        # 6. Define learner
        learner = Learner(scheme, target_steps=args.num_env_steps, log_dir=args.log_dir)

        # 7. Define train loop
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

            if os.path.exists(os.path.join(args.log_dir, "reward_demos/found_demo.npz")):
                break

            iterations += 1

        print("Finished!")
        sys.exit()


def get_args():
    parser = argparse.ArgumentParser(description="RL")

    # Configuration file, keep first
    parser.add_argument("--conf", "-c", type=open, action=LoadFromFile)

    # Wandb
    parser.add_argument(
        "--experiment_name", default=None, help="Name of the wandb experiment the agent belongs to")
    parser.add_argument(
        "--agent_name", default=None, help="Name of the wandb run")
    parser.add_argument(
        "--wandb_key", default=None, help="Init key from wandb account")

    # Environment specs
    parser.add_argument(
        "--env-id", type=str, default=None, help="Gym environment id (default None)")
    parser.add_argument(
        "--frame-skip", type=int, default=0,
        help="Number of frame to skip for each action (default no skip)")
    parser.add_argument(
        "--frame-stack", type=int, default=0,
        help="Number of frame to stack in observation (default no stack)")
    parser.add_argument(
        "--clip_rewards", action="store_true", default=False,
        help="Clip environment rewards")
    parser.add_argument(
        "--episodic_life", action="store_true", default=False,
        help="Turn every life into an episode")

    # RND PPOD specs
    parser.add_argument(
        "--lr", type=float, default=7e-4, help="learning rate (default: 7e-4)")
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="discount factor for rewards (default: 0.99)")
    parser.add_argument(
        "--rho", type=float, default=0.3,
        help="PPO+D rho parameter (default: 0.3)")
    parser.add_argument(
        "--phi", type=float, default=0.0,
        help="PPO+D phi parameter (default: 0.0)")
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95,
        help="gae lambda parameter (default: 0.95)")
    parser.add_argument(
        "--entropy-coef", type=float, default=0.01,
        help="entropy term coefficient (default: 0.01)")
    parser.add_argument(
        "--value-loss-coef", type=float, default=0.5,
        help="value loss coefficient (default: 0.5)")
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5,
        help="max norm of gradients (default: 0.5)")
    parser.add_argument(
        "--use_clipped_value_loss", action="store_true", default=False,
        help="clip value loss update")
    parser.add_argument(
        "--num-steps", type=int, default=20000,
        help="number of forward steps in PPO (default: 20000)")
    parser.add_argument(
        "--ppo-epoch", type=int, default=4,
        help="number of ppo epochs (default: 4)")
    parser.add_argument(
        "--num-mini-batch", type=int, default=32,
        help="number of batches for ppo (default: 32)")
    parser.add_argument(
        "--clip-param", type=float, default=0.2,
        help="ppo clip parameter (default: 0.2)")
    parser.add_argument(
        "--gamma-intrinsic", type=float, default=0.99,
        help="rnd ppo intrinsic gamma parameter (default: 0.99)")
    parser.add_argument(
        "--ext-adv-coeff", type=float, default=2.0,
        help="rnd ppo external advantage coefficient parameter (default: 2.0)")
    parser.add_argument(
        "--int-adv-coeff", type=float, default=1.0,
        help="rnd ppo internal advantage coefficient parameter (default: 1.0)")
    parser.add_argument(
        "--predictor-proportion", type=float, default=1.0,
        help="rnd ppo proportion of batch samples to use to update predictor net (default: 1.0)")
    parser.add_argument(
        "--pre-normalization-steps", type=int, default=50,
        help="rnd ppo number of pre-normalization steps parameter (default: 50)")
    parser.add_argument(
        "--initial-reward-threshold", type=float, default=1.0,
        help="initial reward threshold to add a demo to the replay buffer (default: 1.0)")

    # Feature extractor model specs
    parser.add_argument(
        "--feature-extractor-net", default="MLP", help="Type of nn. Options include MLP, CNN, Fixup")
    parser.add_argument(
        "--restart-model", default=None,
        help="Restart training using the given model")
    parser.add_argument(
        "--restart-reference-model", default=None,
        help="Restart training using the given reference model")
    parser.add_argument(
        "--recurrent-net", default=None, help="Recurrent neural networks to use")
    parser.add_argument(
        '--shared-policy-value-network', action='store_true', default=False,
        help='Shared feature extractor for value network and policy')

    # Scheme specs
    parser.add_argument(
        "--num-env-processes", type=int, default=16,
        help="how many training CPU processes to use (default: 16)")
    parser.add_argument(
        "--num-grad-workers", type=int, default=1,
        help="how many agent workers to use (default: 1)")
    parser.add_argument(
        "--com-grad-workers", default="synchronous",
        help="communication patters grad workers (default: synchronous)")
    parser.add_argument(
        "--num-col-workers", type=int, default=1,
        help="how many agent workers to use (default: 1)")
    parser.add_argument(
        "--com-col-workers", default="synchronous",
        help="communication patters col workers (default: synchronous)")
    parser.add_argument(
        "--cluster", action="store_true", default=False,
        help="script is running in a cluster")

    # General training specs
    parser.add_argument(
        "--num-env-steps", type=int, default=10e7,
        help="number of environment steps to train (default: 10e6)")
    parser.add_argument(
        "--max-time", type=int, default=-1,
        help="stop script after this amount of time in seconds (default: no limit)")
    parser.add_argument(
        "--log-interval", type=int, default=1,
        help="log interval, one log per n updates (default: 10)")
    parser.add_argument(
        "--save-interval", type=int, default=100,
        help="save interval, one save per n updates (default: 100)")
    parser.add_argument(
        "--log-dir", default="/tmp/minigrid_ppo",
        help="directory to save agent logs (default: /tmp/minigrid_ppo)")

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args


if __name__ == "__main__":
    main()
