#!/usr/bin/env python3

import os
import ray
import sys
import time
import wandb
import torch
import argparse

import pytorchrl as prl
from pytorchrl.learner import Learner
from pytorchrl.scheme import Scheme
from pytorchrl.agent.algorithms import PPO
from pytorchrl.agent.algorithms.policy_loss_addons import AttractionKL
from pytorchrl.agent.env import VecEnv
from pytorchrl.agent.storages import GAEBuffer
from pytorchrl.utils import LoadFromFile, save_argparse, cleanup_log_dir
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor, get_memory_network
from pytorchrl.envs.generative_chemistry.utils import adapt_checkpoint
from pytorchrl.envs.generative_chemistry.vocabulary import SMILESTokenizer, create_vocabulary
from pytorchrl.envs.generative_chemistry.generative_chemistry_env_factory import generative_chemistry_train_env_factory

# Default scoring function. Can be replaced by any other scoring function that accepts a SMILE and returns a score!
from pytorchrl.envs.generative_chemistry.default_scoring_function import scoring_function


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

        # 0. Load REINVENT pretrained checkpoint
        vocabulary, tokenizer, max_sequence_length, network_params, network_weights = adapt_checkpoint(os.path.join(
            os.path.dirname(__file__), '../../../pytorchrl/envs/generative_chemistry/models/random.prior.new'))
        restart_model = {"policy_net": network_weights}
        network_params.pop("cell_type")
        network_params.pop("embedding_layer_size")

        # 1. Define Train Vector of Envs
        info_keywords = ("molecule", )
        info_keywords += (
            "regression_model",
            "matching_substructure",
            "custom_alerts",
            "QED_score",
            "raw_regression_model",
            "valid_smiles"
        )

        train_envs_factory, action_space, obs_space = VecEnv.create_factory(
            env_fn=generative_chemistry_train_env_factory,
            env_kwargs={
                "scoring_function": scoring_function,
                "tokenizer": tokenizer, "vocabulary": vocabulary,
                "smiles_max_length": max_sequence_length,
            },
            vec_env_size=args.num_env_processes, log_dir=args.log_dir,
            info_keywords=info_keywords)

        # 2. Define RL Policy
        actor_factory = OnPolicyActor.create_factory(
            obs_space, action_space, prl.PPO,
            feature_extractor_network=get_feature_extractor(args.feature_extractor_net),
            feature_extractor_kwargs={"vocabulary_size": len(vocabulary)},
            recurrent_net=get_memory_network(args.recurrent_net),
            recurrent_net_kwargs={**network_params},
            restart_model=restart_model,
        )

        # 3. Define RL training algorithm
        prior_similarity_addon = AttractionKL(
            behavior_factories=[actor_factory],
            behavior_weights=[1.0],
            loss_term_weight=args.kl_coef,
        )
        algo_factory, algo_name = PPO.create_factory(
            lr=args.lr, eps=args.eps, num_epochs=args.ppo_epoch, clip_param=args.clip_param,
            entropy_coef=args.entropy_coef, value_loss_coef=args.value_loss_coef,
            max_grad_norm=args.max_grad_norm, num_mini_batch=args.num_mini_batch,
            use_clipped_value_loss=args.use_clipped_value_loss, gamma=args.gamma,
            policy_loss_addons=[prior_similarity_addon]
        )

        # 4. Define rollouts storage
        storage_factory = GAEBuffer.create_factory(size=args.num_steps, gae_lambda=args.gae_lambda)

        # 5. Define scheme
        params = {
            "algo_factory": algo_factory,
            "actor_factory": actor_factory,
            "storage_factory": storage_factory,
            "train_envs_factory": train_envs_factory,
        }

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

            iterations += 1

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
        '--agent-name', default=None, help='Name of the wandb run')
    parser.add_argument(
        '--wandb-key', default=None, help='Init key from wandb account')

    # Environment specs
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action (default no skip)')
    parser.add_argument(
        '--frame-stack', type=int, default=0,
        help='Number of frame to stack in observation (default no stack)')

    # PPO specs
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps', type=float, default=1e-5,
        help='Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae', action='store_true', default=False,
        help='use generalized advantage estimation')
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
    parser.add_argument(
        '--kl-coef', type=float, default=0.5,
        help='discount factor for rewards (default: 0.5)')

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
        '--log-dir', default='/tmp/obstacle_tower_ppo',
        help='directory to save agent logs (default: /tmp/obstacle_tower_ppo)')

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args


if __name__ == "__main__":
    main()
