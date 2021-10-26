#!/usr/bin/env python3

import glob
import torch
import random
import numpy as np
import pytorchrl as prl
from pytorchrl.agent.env import VecEnv
from pytorchrl.envs.obstacle_tower.obstacle_tower_env_factory import obstacle_train_env_factory
from code_examples.train_obstacle_tower.ppod.train import get_args


def enjoy():

    args = get_args()

    # Define single copy of the environment
    env, _, _ = VecEnv.create_factory(
        env_fn=obstacle_train_env_factory,
        env_kwargs={
            "frame_skip": args.frame_skip,
            "reward_shape": args.reward_shape,
            "reduced_actions": args.reduced_action_space,
            "num_actions": args.num_actions,
            "realtime": True,
        },
        vec_env_size=1)

    # Define agent device and agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Start recording
    env = env()

    demos_list = glob.glob(args.demos_dir + '/*.npz')
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
