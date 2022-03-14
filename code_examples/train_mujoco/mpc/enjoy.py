#!/usr/bin/env python3

import os
import torch
import argparse
from pytorchrl.envs.pybullet import pybullet_test_env_factory
from pytorchrl.agent.actors import MBActor
from pytorchrl.agent.algorithms import MB_MPC
from pytorchrl.utils import LoadFromFile


def enjoy():

    args = get_args()

    # Define single copy of the environment
    env = pybullet_test_env_factory(env_id=args.env_id)
    env.render()

    # Define agent device and agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dynamics_model =  MBActor.create_factory(args.env_id,
                                                  env.observation_space,
                                                  env.action_space,
                                                  hidden_size=args.hidden_size,
                                                  batch_size=args.mini_batch_size,
                                                  learn_reward_function=args.learn_reward_function,
                                                  checkpoint=os.path.join(args.log_dir, "model.state_dict"))(device)

    algo_factory, algo = MB_MPC.create_factory(args)
    
    mpc = algo_factory(device=device,
                       actor=dynamics_model,
                       envs=env)
    
    # Define initial Tensors
    obs, done = env.reset(), False
    _, rhs, _ = dynamics_model.actor_initial_states(torch.tensor(obs))
    episode_reward = 0

    # Execute episodes
    while not done:

        env.render()
        obs = torch.Tensor(obs).view(1, -1).to(device)
        done = torch.Tensor([done]).view(1, -1).to(device)
        with torch.no_grad():
            _, clipped_action, rhs, _ = mpc.acting_step(obs, rhs, done, deterministic=True)
        obs, reward, done, info = env.step(clipped_action.squeeze().cpu().numpy())
        episode_reward += reward

        if done:
            print("EPISODE: reward: {}".format(episode_reward), flush=True)
            done, episode_reward = 0, False
            obs = env.reset()


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # Configuration file, keep first
    parser.add_argument('--conf','-c', type=open, action=LoadFromFile)
    
    # Wandb
    parser.add_argument(
        '--experiment_name', default=None,
        help='Name of the wandb experiment the agent belongs to')
    parser.add_argument(
        '--agent_name', default=None, help='Name of the wandb run')
    parser.add_argument(
        '--wandb_key', default=None, help='Init key from wandb account')

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
        "--learn-reward-function", default=False, action='store_true', help="")
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--mb_epochs', type=int, default=60,
        help='Number of epochs to train the dynamics model, (default: 60)')
    parser.add_argument(
        '--eps', type=float, default=1e-8,
        help='Adam optimizer epsilon (default: 1e-8)')
    parser.add_argument(
        "--n-planner", type=int, default=1000,
        help="Number of parallel planner for each actor (default: 500)")
    parser.add_argument(
        "--horizon", type=int, default=12,
        help="The horizon of online planning (default: 12)")
    parser.add_argument(
        "--mpc-type", type=str, choices=["RS", "CEM", "PDDM"], default="RS",
        help="Type of MPC optimizer, RS: Random Shooting, CEM: Cross Entropy Method (default: RS)")
    parser.add_argument(
        "--action-noise", default=False, action='store_true',
        help="Adding noise to the actions, (default: False)")
    parser.add_argument(
        '--start-steps', type=int, default=5000,
        help='SAC num initial random steps (default: 1000)')
    parser.add_argument(
        '--buffer-size', type=int, default=10000,
        help='Rollouts storage size (default: 10000 transitions)')
    parser.add_argument(
        '--update-every', type=int, default=50,
        help='Num env collected steps between dynamics network update stages (default: 50)')
    parser.add_argument(
        '--num-updates', type=int, default=50,
        help='Num network updates per dynamics network update stage (default 50)')
    parser.add_argument(
        '--mini-batch-size', type=int, default=32,
        help='Mini batch size for network updates (default: 32)')
    parser.add_argument("--test_every", type=int, default=1, help="")
    
    # CEM parameter
    parser.add_argument(
        "--iter-update-steps", type=int, default=3,
        help="Iterative update steps for CEM (default: 3)")
    parser.add_argument(
        "--k-best", type=int, default=5, 
        help="K-Best members of the mean prediction forming the next mean distribution")
    parser.add_argument(
        "--update-alpha", type=float, default=0.0,
        help="Soft update alpha for each iteration (default: 0.0)")

    # PDDM parameter
    parser.add_argument(
        "--gamma", type=float, default=1.0,
        help="PDDM gamma value (default: 1.0)")
    parser.add_argument(
        "--beta", type=float, default=0.5,
        help="PDDM beta value (default: 0.5)")

    # Feature dynamics model specs
    parser.add_argument(
        "--hidden-size", type=int, default=256,
        help="Number of hidden nodes for the dynamics model (default: 256)")
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
    enjoy()