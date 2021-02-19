#!/usr/bin/env python3

import os
import argparse
from glob import glob
from baselines.bench import load_results
from matplotlib import pylab as plt; plt.rcdefaults()
from pytorchrl.utils import colorize, LoadFromFile



def plot(experiment_path, roll=5, save_name="results"):

    fig = plt.figure(figsize=(20, 10))

    if len(glob(os.path.join(experiment_path, "train/*monitor*"))) != 0:

        exps = glob(experiment_path)
        print(exps)

        # Get data
        df = load_results(os.path.join(experiment_path, "train"))
        rdf = df.rolling(roll)
        df['steps'] = df['l'].cumsum()
        if 'rrr' in df:
            df = df[df['lives'] == 0]
            df['r'] = df['rrr']

        ax = plt.subplot(1, 1, 1)
        df.rolling(roll).mean().plot('steps', 'r',  style='-',  ax=ax,  legend=False)
        rdf.max().plot('steps', 'r', style='-', ax=ax, legend=False, color="#28B463", alpha=0.65)
        rdf.min().plot('steps', 'r', style='-', ax=ax, legend=False, color="#F39C12", alpha=0.65)

        # X axis
        ax.set_xlabel('Num steps (M)')

        # Y axis
        ax.set_ylabel('Reward')
        ax.grid(True)

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.2)

    # Save figure
    save_name = os.path.join(experiment_path, save_name) + ".jpg"
    ax.get_figure().savefig(save_name)
    print(colorize("Plot save as: {}".format(save_name), color="green"))
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