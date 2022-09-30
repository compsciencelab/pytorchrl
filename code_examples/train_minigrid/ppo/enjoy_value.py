#!/usr/bin/env python3

import os
import time
import torch
import argparse
import pytorchrl as prl
from minigrid.utils.window import Window
from pytorchrl.envs.minigrid.minigrid_env_factory import minigrid_test_env_factory
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor
from pytorchrl.utils import LoadFromFile
from pytorchrl.agent.env.env_wrappers import TransposeImagesIfRequired
from code_examples.train_minigrid.ppo.train import get_args


def redraw(window, img):
    window.show_img(img)


def reset(env, window, seed=None):
    env.reset(seed=seed)

    if hasattr(env, "mission"):
        print("Mission: %s" % env.mission)
        window.set_caption(env.mission)

    img = env.get_frame()

    redraw(window, img)


def step(env, window, action, policy):
    obs, reward, done, info = env.step(action)

    # Define tensors
    device = policy.device
    obs = torch.Tensor(obs).unsqueeze(0).to(device)
    done = torch.Tensor([done]).unsqueeze(0).to(device)
    _, rhs, _ = policy.actor_initial_states(obs)

    with torch.no_grad():
        _, _, _, rhs, _, _ = policy.get_action(obs, rhs, done, deterministic=False)
        value = policy.get_value(obs, rhs, done)['value_net1']

    print(f"step={env.step_count}, reward={reward:.2f}, value={value.item():.2f}")

    if done:
        print("terminated!")
        reset(env, window)
    else:
        img = env.get_frame()
        redraw(window, img)


def key_handler(env, window, event, policy):
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset(env, window)
        return

    if event.key == "left":
        step(env, window, env.actions.left, policy)
        return
    if event.key == "right":
        step(env, window, env.actions.right, policy)
        return
    if event.key == "up":
        step(env, window, env.actions.forward, policy)
        return

    # Spacebar
    if event.key == "t":
        step(env, window, env.actions.toggle, policy)
        return
    if event.key == "k":
        step(env, window, env.actions.pickup, policy)
        return
    if event.key == "d":
        step(env, window, env.actions.drop, policy)
        return

    if event.key == "enter":
        step(env, window, env.actions.done, policy)
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
        restart_model=os.path.join(args.log_dir, "model.state_dict"))(device)

    # Define initial Tensors
    obs, done = env.reset(), False
    _, rhs, _ = policy.actor_initial_states(torch.tensor(obs))

    # Execute episodes
    window = Window("minigrid - MiniGrid-DeceivingRewards-v0")
    window.reg_key_handler(lambda event: key_handler(env, window, event, policy))

    seed = None
    reset(env, window, seed)

    # Blocking event loop
    window.show(block=True)


if __name__ == "__main__":
    enjoy()
