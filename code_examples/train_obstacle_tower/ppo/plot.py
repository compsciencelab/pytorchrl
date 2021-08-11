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

    if len(glob(os.path.join(experiment_path, "train/*monitor*"))) != 0:

        exps = glob(experiment_path)
        print(exps)

        df_train = load_baselines_results(os.path.join(experiment_path, "train"))
        df_train['steps'] = df_train['l'].cumsum() / 1000000
        df_train['time'] = df_train['t'] / 3600

        ax = plt.subplot(1, 1, 1)
        df_train.rolling(roll).mean().plot('steps', 'r',  style='-',  ax=ax,  legend=False)

    if len(glob(os.path.join(experiment_path, "test/*monitor*"))) != 0:

        exps = glob(experiment_path)
        print(exps)

        df_test = load_baselines_results(os.path.join(experiment_path, "test"))
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

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)

    return args

if __name__ == "__main__":

    args = get_args()
    plot(experiment_path=args.log_dir, save_name=args.save_name)
    quit()