#!/usr/bin/env python3

import os
import time
import torch
import argparse
import numpy as np
import pytorchrl as prl
from minigrid.utils.window import Window
from pytorchrl.envs.minigrid.minigrid_env_factory import minigrid_test_env_factory
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor
from pytorchrl.utils import LoadFromFile
from pytorchrl.agent.env.env_wrappers import TransposeImagesIfRequired
from code_examples.train_minigrid.ppo.train import get_args


class EnvManager:

    def __init__(self, env, window, policy):

        self.env = env
        self.policy = policy
        self.window = window
        self.value = 0.0

    def redraw(self, img):
        self.window.show_img(img)

    def reset(self, seed=None):
        self.env.reset(seed=seed)

        if hasattr(self.env, "mission"):
            print("Mission: %s" % self.env.mission)
            self.window.set_caption(self.env.mission)

        img = self.env.get_frame()
        self.redraw(img)

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        value_error = np.abs(self.value - reward)
        print(f"step={self.env.step_count}, reward={reward:.2f}, value={self.value:.2f}, value_error={value_error:.2f}")

        # Define tensors
        device = self.policy.device
        obs = torch.Tensor(obs).unsqueeze(0).to(device)
        done = torch.Tensor([done]).unsqueeze(0).to(device)
        _, rhs, _ = self.policy.actor_initial_states(obs)

        with torch.no_grad():
            self.value = self.policy.get_value(obs, rhs, done)['value_net1'].item()

        if done:
            print("terminated!")
            self.reset()
        else:
            img = self.env.get_frame()
            self.redraw(img)


def key_handler(manager, event):
    print("pressed", event.key)

    if event.key == "escape":
        manager.window.close()
        return

    if event.key == "backspace":
        manager.reset()
        return

    if event.key == "left":
        manager.step(manager.env.actions.left)
        return
    if event.key == "right":
        manager.step(manager.env.actions.right)
        return
    if event.key == "up":
        manager.step(manager.env.actions.forward)
        return

    # Spacebar
    if event.key == "t":
        manager.step(manager.env.actions.toggle)
        return
    if event.key == "k":
        manager.step(manager.env.actions.pickup)
        return
    if event.key == "d":
        manager.step(manager.env.actions.drop)
        return

    if event.key == "enter":
        manager.step(manager.env.actions.done)
        return


def enjoy():

    args = get_args()

    # Define single copy of the environment
    env = minigrid_test_env_factory(env_id=args.env_id)
    env = TransposeImagesIfRequired(env)

    # Define agent device and agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    policy = OnPolicyActor.create_factory(
        env.observation_space, env.action_space, prl.PPO,
        restart_model=os.path.join(args.log_dir, "model.state_dict"),
        shared_policy_value_network=False,
    )(device)

    # Execute episodes
    window = Window("minigrid - MiniGrid-DeceivingRewards-v0")
    manager = EnvManager(env, window, policy)
    window.reg_key_handler(lambda event: key_handler(manager, event))

    seed = None
    manager.reset(seed)

    # Blocking event loop
    window.show(block=True)


if __name__ == "__main__":
    enjoy()
