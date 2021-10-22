#!/usr/bin/env python3

import glob
import random
import numpy as np
import pytorchrl as prl
from matplotlib import pyplot as plt
from code_examples.train_obstacle_tower.ppod.train import get_args


def visualize():

    args = get_args()
    demos_list = glob.glob(args.demos_dir + '/*.npz')
    demo_name = random.choice(demos_list)
    demo = np.load(demo_name)
    done, step = False, 0
    length_demo = demo[prl.ACT].shape[0]
    print("LOADING DEMO: {}, LENGTH {}".format(demo_name, length_demo))

    fig = plt.figure()

    # Execute episodes
    while not done:

        fig.clf()
        plt.imshow(np.moveaxis(demo[prl.OBS][step], [0, 1, 2], [2, 0, 1]) / 255.)
        plt.draw()
        plt.pause(0.02)
        step += 1

        if step == length_demo:
            step = 0
            demo_name = random.choice(demos_list)
            demo = np.load(demo_name)
            length_demo = demo[prl.ACT].shape[0]
            print("LOADING DEMO: {}, LENGTH {}".format(demo_name, length_demo))


if __name__ == "__main__":
    visualize()
