#!/usr/bin/env python3

import os
import argparse
import numpy as np
from glob import glob
from matplotlib import pylab as plt; plt.rcdefaults()
from pytorchrl.agent.env import load_baselines_results
from pytorchrl.utils import LoadFromFile


def plot(experiment_path, roll=50, save_name="results"):

    fig = plt.figure(figsize=(20, 10))

    if len(glob(os.path.join(experiment_path, "monitor_logs/train/*monitor*"))) != 0:

        exps = glob(experiment_path)
        print(exps)

        df_train = load_baselines_results(os.path.join(experiment_path, "monitor_logs/train"))
        df_train['steps'] = df_train['l'].cumsum() / 1000000
        df_train['time'] = df_train['t'] / 3600

        ax = plt.subplot(1, 1, 1)
        df_train.rolling(roll).mean().plot('steps', 'r',  style='-',  ax=ax,  legend=False)

    if len(glob(os.path.join(experiment_path, "monitor_logs/test/*monitor*"))) != 0:

        exps = glob(experiment_path)
        print(exps)

        df_test = load_baselines_results(os.path.join(experiment_path, "monitor_logs/test"))
        df_test['steps'] = df_test['l'].cumsum() / 1000000
        df_test['time'] = df_test['t'] / 3600

        # Map test steps with corresponding number of training steps
        df_test["steps"] = df_test["steps"].map(
            lambda a: df_train["steps"][np.argmin(abs(df_test["time"][df_test["steps"].index[
                df_test["steps"] == a]].to_numpy() - df_train["time"]))])

        ax = plt.subplot(1, 1, 1)
        df_test.rolling(roll).mean().plot('steps', 'r', style='-', ax=ax, legend=False)

    fig.legend(["train", "test"], loc="lower center", ncol=2)
    ax.set_title("Unity3D Obstacle Tower Environment")
    ax.set_xlabel('Num steps (M)')
    ax.set_ylabel('Reward')
    ax.grid(True)

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.2)

    # Save figure
    save_name = os.path.join(experiment_path, save_name) + ".jpg"
    ax.get_figure().savefig(save_name)
    print("Plot saved as: {}".format(save_name))
    plt.clf()


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # Configuration file
    parser.add_argument('--conf','-c', type=open, action=LoadFromFile)

    parser.add_argument(
        '--save-name', default='results',
        help='plot save name (default: results)')

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
        '--log-dir', default='/tmp/obstacle_tower_ppo',
        help='directory to save agent logs (default: /tmp/obstacle_tower_ppo)')

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)

    return args


if __name__ == "__main__":

    args = get_args()
    plot(experiment_path=args.log_dir, save_name=args.save_name)
    quit()