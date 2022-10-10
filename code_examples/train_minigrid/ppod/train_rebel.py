#!/usr/bin/env python3

import os
import sys
import time
import json
import wandb
import argparse
import numpy as np

from pytorchrl.learner import Learner
from pytorchrl.scheme import Scheme
from pytorchrl.agent.algorithms import PPO
from pytorchrl.agent.env import VecEnv
from pytorchrl.agent.storages import PPOD2RebelBuffer
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor
from pytorchrl.envs.minigrid.minigrid_env_factory import minigrid_train_env_factory
from pytorchrl.utils import LoadFromFile, save_argparse, cleanup_log_dir
from code_examples.train_minigrid.ppod.train import get_args


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
        algo_factory, algo_name = PPO.create_factory(
            lr=args.lr, num_epochs=args.ppo_epoch, clip_param=args.clip_param,
            entropy_coef=args.entropy_coef, value_loss_coef=args.value_loss_coef,
            max_grad_norm=args.max_grad_norm, num_mini_batch=args.num_mini_batch,
            use_clipped_value_loss=args.use_clipped_value_loss, gamma=args.gamma,)

        # 3. Define RL Policy
        actor_factory = OnPolicyActor.create_factory(
            obs_space, action_space, algo_name,
            restart_model=args.restart_reference_model,
            shared_policy_value_network=args.shared_policy_value_network)

        # 4. Define rollouts storage
        storage_factory = PPOD2RebelBuffer.create_factory(
            size=args.num_steps, gae_lambda=args.gae_lambda,
            general_value_net_factory=actor_factory,
            target_reward_demos_dir=os.path.join(args.log_dir, "reward_demos"),
            initial_reward_threshold=1.0)

        actor_factory = OnPolicyActor.create_factory(
            obs_space, action_space, algo_name,
            restart_model=args.restart_model,
            shared_policy_value_network=args.shared_policy_value_network)

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


if __name__ == "__main__":
    main()
