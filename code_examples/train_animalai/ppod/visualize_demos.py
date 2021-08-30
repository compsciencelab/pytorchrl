#!/usr/bin/env python3

import os
import glob
import torch
import random
import numpy as np
import pytorchrl as prl
from pytorchrl.agent.env import VecEnv
from pytorchrl.envs.animal_olympics.animal_olympics_env_factory import animal_train_env_factory
from code_examples.train_animalai.ppod.train import get_args


def enjoy():

    args = get_args()
    args.path_to_demos_dir = "/tmp/demos/"

    # Define single copy of the environment
    arena_file = os.path.dirname(os.path.abspath(__file__)) + "/arenas/"
    env, action_space, obs_space = VecEnv.create_factory(
        env_fn=animal_train_env_factory,
        env_kwargs={
            "arenas_dir": arena_file,
            "frame_skip": args.frame_skip,
            "frame_stack": args.frame_stack,
            "inference": True,
        }, vec_env_size=1)

    # Define agent device and agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Start recording
    env = env()
    obs = env.reset()

    demos_list = glob.glob(args.path_to_demos_dir + '/*.npz')
    demo_name = random.choice(demos_list)
    demo = np.load(demo_name)
    done, episode_reward, step = False, 0, 0
    length_demo = demo[prl.ACT].shape[0]
    print("LOADING DEMO: {}, LENGTH {}".format(demo_name, length_demo))

    # Execute episodes
    while not done:

        obs, reward, done, info = env.step(torch.Tensor(demo[prl.ACT][step]).view(1, -1).to(device))
        episode_reward += reward
        step += 1
        print(step)

        if step == length_demo:
            done = True

        if done:
            print("EPISODE: reward: {}".format(episode_reward.item()), flush=True)
            done, episode_reward, step = False, 0, 0
            obs = env.reset()
            demo_name = random.choice(demos_list)
            demo = np.load(demo_name)
            length_demo = demo[prl.ACT].shape[0]
            print("LOADING DEMO: {}, LENGTH {}".format(demo_name, length_demo))


if __name__ == "__main__":
    enjoy()
