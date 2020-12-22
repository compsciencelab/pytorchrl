import numpy as np
from glob import glob
from matplotlib import pylab as plt; plt.rcdefaults()
from baselines.bench import load_results


def plot(experiment_path, save_dir="/tmp/", save_name="results", limit_steps=None):

    fig = plt.figure(figsize=(20, 10))

    if len(glob(os.path.join(experiment_path, "train/*monitor*"))) != 0:

        exps = glob(experiment_path)
        print(exps)

        # Get data
        df = load_results(os.path.join(experiment_path, "train"))
        roll = 5
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
        gap = 1
        ax.set_xticks(np.arange(0, ((df['steps'].iloc[-1] // gap) + 1) * gap, gap))
        ax.set_xlabel('Num steps (M)')
        if limit_steps: plt.xlim(0, limit_steps)

        # Y axis
        gap = 25
        ax.set_yticks(np.arange(((df['r'].min() // gap) - 1) * gap, ((df['r'].max() // gap) + 1) * gap, gap))
        ax.set_ylabel('Reward')
        ax.grid(True)

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.2)

    # Save figure
    ax.get_figure().savefig(os.path.join(save_dir, save_name) + ".jpg")
    plt.clf()


if __name__ == "__main__":

    import os
    import argparse

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--log-dir', default='/tmp/',
        help='experiment directory or directory containing '
             'several experiments (default: /tmp/ppo)')
    parser.add_argument(
        '--save-dir', default='/tmp/',
        help='path to desired save directory (default: /tmp/)')
    parser.add_argument(
        '--save-name', default='results',
        help='plot save name (default: results)')
    parser.add_argument(
        '--black-list', action='store', type=str, nargs='*', default=[],
        help="experiments to be ignored. Example: -i item1 item2 -i item3 "
             "(default [])")
    parser.add_argument(
        '--limit-steps', type=int, default=None,
        help='truncate plots at this number of steps (default: None)')
    parser.add_argument(
        '--min-max', action='store_true', default=True,
        help='whether or not to plot rolling window min and max values')

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)

    args.log_dir = "/tmp/atari_sac/"
    args.save_dir = "/tmp"

    plot(experiment_path=args.log_dir,
         save_dir=args.save_dir, save_name=args.save_name,
         limit_steps=args.limit_steps)

    quit()