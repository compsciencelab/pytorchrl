#!/usr/bin/env python3

import glob
import numpy as np
import pytorchrl as prl
from matplotlib import pyplot as plt
from code_examples.train_obstacle_tower.ppod.train import get_args


def visualize():

    args = get_args()

    demo_idx = 0
    demos_list = sorted(glob.glob(args.demos_dir + '/*.npz'))
    demo_name = demos_list[demo_idx]
    demo = np.load(demo_name)
    done, episode_reward, step = False, 0, 0
    length_demo = demo[prl.ACT].shape[0]
    print("LOADING DEMO: {}, LENGTH {}".format(demo_name, length_demo))

    fig = plt.figure()

    # Execute episodes
    while not done:

        try:

            fig.clf()
            plt.imshow(np.moveaxis(demo[prl.OBS][step], [0, 1, 2], [2, 0, 1]) / 255.)
            plt.draw()
            plt.pause(0.02)
            step += 1

            if step == length_demo:
                print("EPISODE: reward: {}\n".format(demo[prl.REW].sum()))
                step = 0
                demo_idx += 1
                if demo_idx == len(demos_list):
                    break
                demo_name = demos_list[demo_idx]
                demo = np.load(demo_name)
                length_demo = demo[prl.ACT].shape[0]
                print("LOADING DEMO: {}, LENGTH {}".format(demo_name, length_demo))

        except KeyboardInterrupt:
            step = 0
            demo_idx += 1
            if demo_idx == len(demos_list):
                break
            demo_name = demos_list[demo_idx]
            demo = np.load(demo_name)
            length_demo = demo[prl.ACT].shape[0]
            print("LOADING DEMO: {}, LENGTH {}, REWARD {}".format(
                demo_name, length_demo, demo[prl.REW].sum()))

    print("Finished!")


if __name__ == "__main__":
    visualize()
