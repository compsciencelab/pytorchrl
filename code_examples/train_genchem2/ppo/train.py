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
from pytorchrl.envs.generative_chemistry.vocabulary import SMILESTokenizer, create_vocabulary
from pytorchrl.envs.generative_chemistry2.generative_chemistry_env_factory import generative_chemistry_train_env_factory

# TODO: update this dict
weights_mapping = {
    "_embedding.weight": "policy_net.memory_net._embedding.weight",
    "_rnn.weight_ih_l0": "policy_net.memory_net._rnn.weight_ih_l0",
    "_rnn.weight_hh_l0": "policy_net.memory_net._rnn.weight_hh_l0",
    "_rnn.bias_ih_l0": "policy_net.memory_net._rnn.bias_ih_l0",
    "_rnn.bias_hh_l0": "policy_net.memory_net._rnn.bias_hh_l0",
    "_rnn.weight_ih_l1": "policy_net.memory_net._rnn.weight_ih_l1",
    "_rnn.weight_hh_l1": "policy_net.memory_net._rnn.weight_hh_l1",
    "_rnn.bias_ih_l1": "policy_net.memory_net._rnn.bias_ih_l1",
    "_rnn.bias_hh_l1": "policy_net.memory_net._rnn.bias_hh_l1",
    "_rnn.weight_ih_l2": "policy_net.memory_net._rnn.weight_ih_l2",
    "_rnn.weight_hh_l2": "policy_net.memory_net._rnn.weight_hh_l2",
    "_rnn.bias_ih_l2": "policy_net.memory_net._rnn.bias_ih_l2",
    "_rnn.bias_hh_l2": "policy_net.memory_net._rnn.bias_hh_l2",
    "_linear.weight": "policy_net.dist.linear.weight",
    "_linear.bias": "policy_net.dist.linear.bias",
}


def adapt_checkpoint(file_path):

    if torch.cuda.is_available():
        save_dict = torch.load(file_path)
    else:
        save_dict = torch.load(file_path, map_location=lambda storage, loc: storage)

    new_save_dict = {}

    # Change network weight names
    for k in save_dict["network"].keys():
        new_save_dict[weights_mapping[k]] = save_dict["network"][k]

    # Temporarily save network weight to /tmp/network_params
    torch.save(new_save_dict, "/tmp/network_params.tmp")

    return save_dict['vocabulary'], save_dict['tokenizer'], save_dict['max_sequence_length'],\
           save_dict['network_params'], "/tmp/network_params.tmp"


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

        # 0. Load checkpoint
        try:
            vocabulary, tokenizer, max_sequence_length, network_params, network_weights = adapt_checkpoint(
                os.path.join(os.path.dirname(__file__), '../../../pytorchrl/envs/generative_chemistry2/models/random.prior.new'))
            restart_model = {"policy_net": network_weights}
            smiles_list = []
        except Exception:
            vocabulary, tokenizer, max_sequence_length, network_params, network_weights = None, None, 100, {}, None
            restart_model = None
            smiles_list = ["[*:0]N1CCN(CC1)CCCCN[*:1]"]

        # 1. Define Train Vector of Envs
        if tokenizer is None and vocabulary is None:
            tokenizer = SMILESTokenizer()
            vocabulary = create_vocabulary(smiles_list, tokenizer)
            network_weights = None

        diversity_filter_params = {
            "name": "IdenticalMurckoScaffold",  # other options are: "IdenticalTopologicalScaffold", "NoFilter" and "ScaffoldSimilarity" -> use "NoFilter" to disable this feature
            "nbmax": 25,  # the bin size; penalization will start once this is exceeded
            "minscore": 0.4,  # the minimum total score to be considered for binning
            "minsimilarity": 0.4  # the minimum similarity to be placed into the same bin
        }

        scoring_function_parameters = {
            "name": "custom_product",  # this is our default one (alternative: "custom_sum")
            "parallel": False,  # sets whether components are to be executed
            # in parallel; note, that python uses "False" / "True"
            # but the JSON "false" / "true"

            # the "parameters" list holds the individual components
            "parameters": [

                # add component: an activity model
                {
                    "component_type": "predictive_property",  # this is a scikit-learn model, returning
                    # activity values
                    "name": "Regression model",  # arbitrary name for the component
                    "weight": 2,  # the weight ("importance") of the component (default: 1)
                    "specific_parameters": {
                        "model_path": os.path.join(os.path.dirname(__file__), '../../../pytorchrl/envs/generative_chemistry2/models/Aurora_model.pkl'),
                        # absolute model path
                        "scikit": "regression",  # model can be "regression" or "classification"
                        "descriptor_type": "ecfp_counts",  # sets the input descriptor for this model
                        "size": 2048,  # parameter of descriptor type
                        "radius": 3,  # parameter of descriptor type
                        "use_counts": True,  # parameter of descriptor type
                        "use_features": True,  # parameter of descriptor type
                        "transformation": {
                            "transformation_type": "sigmoid",  # see description above
                            "high": 9,  # parameter for sigmoid transformation
                            "low": 4,  # parameter for sigmoid transformation
                            "k": 0.25  # parameter for sigmoid transformation
                        }
                    }
                },

                # add component: enforce the match to a given substructure
                {
                    "component_type": "matching_substructure",
                    "name": "Matching substructure",  # arbitrary name for the component
                    "weight": 1,  # the weight of the component (default: 1)
                    "specific_parameters": {
                        "smiles": ["c1ccccc1CC"]  # a match with this substructure is required
                    }
                },

                # add component: enforce to NOT match a given substructure
                {
                    "component_type": "custom_alerts",
                    "name": "Custom alerts",  # arbitrary name for the component
                    "weight": 1,  # the weight of the component (default: 1)
                    "specific_parameters": {
                        "smiles": [  # specify the substructures (as list) to penalize
                            "[*;r8]",
                            "[*;r9]",
                            "[*;r10]",
                            "[*;r11]",
                            "[*;r12]",
                            "[*;r13]",
                            "[*;r14]",
                            "[*;r15]",
                            "[*;r16]",
                            "[*;r17]",
                            "[#8][#8]",
                            "[#6;+]",
                            "[#16][#16]",
                            "[#7;!n][S;!$(S(=O)=O)]",
                            "[#7;!n][#7;!n]",
                            "C#C",
                            "C(=[O,S])[O,S]",
                            "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
                            "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
                            "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
                            "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
                            "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
                            "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"
                        ]
                    }
                },

                # add component: calculate the QED drug-likeness score (using RDkit)
                {
                    "component_type": "qed_score",
                    "name": "QED Score",  # arbitrary name for the component
                    "weight": 1,  # the weight of the component (default: 1)
                }]
        }
        train_envs_factory, action_space, obs_space = VecEnv.create_factory(
            env_fn=generative_chemistry_train_env_factory,
            env_kwargs={
                "smiles_list": smiles_list,
                "scoring_function_parameters": scoring_function_parameters,
                "tokenizer": tokenizer, "vocabulary": vocabulary,
                "obs_length": max_sequence_length,
            },
            vec_env_size=args.num_env_processes, log_dir=args.log_dir,
            info_keywords=("molecules", ))

        # 2. Define RL Policy
        # TODO. Actor does not allow to choose alternative memory network
        actor_factory = OnPolicyActor.create_factory(
            obs_space, action_space, prl.PPO,
            feature_extractor_network=torch.nn.Identity,
            feature_extractor_kwargs={},
            recurrent_nets_kwargs={"vocabulary": vocabulary,  **network_params},
            restart_model=restart_model,
            recurrent_nets=get_memory_network("LSTM"),
        )

        # 2. Define RL training algorithm
        prior_similarity_addon = AttractionKL(
            behavior_factories=[actor_factory],
            behavior_weights=[1.0],
            loss_term_weight=0.1,
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
        '--nn', default='CNN', help='Type of nn. Options are MLP, CNN, Fixup')
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
        '--log-dir', default='/tmp/obstacle_tower_ppo',
        help='directory to save agent logs (default: /tmp/obstacle_tower_ppo)')

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args


if __name__ == "__main__":
    main()
