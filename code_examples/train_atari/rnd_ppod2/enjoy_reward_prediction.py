#!/usr/bin/env python3

import os
import sys
import glob
import time
import torch
import threading
import numpy as np
from itertools import count
from pynput import keyboard
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import pytorchrl as prl
from pytorchrl.agent.env import VecEnv
from pytorchrl.envs.atari.utils import imdownscale
from pytorchrl.agent.storages import PPOD2RebelBuffer
from pytorchrl.envs.atari import atari_train_env_factory
from code_examples.train_atari.rnd_ppod2.train import get_args
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor

plt.style.use("fivethirtyeight")
pressed_keys = set([])


def create_action():
    action = 0
    if "s" in pressed_keys and "space" in pressed_keys and "a" in pressed_keys:
        print("PRESSED SPACE, A AND S, ACTION 17")
        action = 17
    elif "s" in pressed_keys and "space" in pressed_keys and "a" in pressed_keys:
        # print("PRESSED SPACE, D AND S, ACTION 16")
        action = 16
    elif "w" in pressed_keys and "space" in pressed_keys and "a" in pressed_keys:
        # print("PRESSED SPACE, W AND D, ACTION 15")
        action = 15
    elif "w" in pressed_keys and "space" in pressed_keys and "d" in pressed_keys:
        # print("PRESSED SPACE, W AND D, ACTION 14")
        action = 14
    elif "w" in pressed_keys and "d" in pressed_keys:
        # print("PRESSED W AND D, ACTION 6")
        action = 6
    elif "w" in pressed_keys and "a" in pressed_keys:
        # print("PRESSED W AND A, ACTION 7")
        action = 7
    elif "s" in pressed_keys and "d" in pressed_keys:
        # print("PRESSED S AND D, ACTION 8")
        action = 8
    elif "s" in pressed_keys and "a" in pressed_keys:
        # print("PRESSED S AND A, ACTION 9")
        action = 9
    elif "space" in pressed_keys and "w" in pressed_keys:
        # print("PRESSED SPACE AND W, ACTION 10")
        action = 10
    elif "space" in pressed_keys and "d" in pressed_keys:
        # print("PRESSED SPACE AND D, ACTION 11")
        action = 11
    elif "space" in pressed_keys and "a" in pressed_keys:
        # print("PRESSED SPACE AND A, ACTION 12")
        action = 12
    elif "space" in pressed_keys and "s" in pressed_keys:
        # print("PRESSED SPACE AND S, ACTION 13")
        action = 13
    elif "w" in pressed_keys:
        # print("PRESSED W, ACTION 2")
        action = 2
    elif "s" in pressed_keys:
        # print("PRESSED S, ACTION 5")
        action = 5
    elif "d" in pressed_keys:
        # print("PRESSED D, ACTION 3")
        action = 3
    elif "a" in pressed_keys:
        # print("PRESSED A, ACTION 4")
        action = 4
    elif "space" in pressed_keys:
        # print("PRESSED SPACE, ACTION 1")
        action = 1
    return action


def on_press(key):
    try:
        pressed_keys.add(key.char if hasattr(key, "char") else "space")
    except KeyError:
        pass


def on_release(key):
    try:
        pressed_keys.remove(key.char if hasattr(key, "char") else "space")
    except KeyError:
        pass

def play():

    args = get_args()

    # Define agent device and agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define single copy of the environment
    env, action_space, obs_space = VecEnv.create_factory(
        env_fn=atari_train_env_factory,
        env_kwargs={
            "env_id": args.env_id,
            "frame_stack": args.frame_stack,
            "episodic_life": args.episodic_life,
            "clip_rewards": args.clip_rewards,
        }, vec_env_size=1)
    env = env()
    obs = env.reset()

    # Define policy
    checkpoints = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.state_dict*")))
    if len(checkpoints) > 0:
        checkpoint = checkpoints[-1]
    else:
        checkpoint = None

    policy = OnPolicyActor.create_factory(
        obs_space, action_space, prl.RND_PPO,
        restart_model=checkpoint,
        recurrent_net=args.recurrent_net,
    )(device)

    storage_factory = PPOD2RebelBuffer.create_factory(
        size=args.num_steps, gae_lambda=args.gae_lambda,
        reward_predictor_factory=get_feature_extractor(args.feature_extractor_net),
        reward_predictor_net_kwargs={
            "input_space": env.observation_space,
            "output_sizes": [256, 448, 1],
            "final_activation": False,
        },
        restart_reward_predictor_net=args.restart_reference_model,
        target_reward_demos_dir=os.path.join(args.log_dir, "reward_demos"),
        initial_reward_threshold=1.0,
    )(device, policy, None, env)
    policy.try_load_from_checkpoint()

    step_count = 0

    with keyboard.Listener(on_press=on_press,  on_release=on_release) as listener:

        threading.Thread(target=listener.join)

        while True:

            env.render()
            with torch.no_grad():
                rew_pred = policy.predictor.reward_predictor(obs).item()

            action = create_action()
            obs, reward, done, info = env.step(torch.tensor([action]).unsqueeze(1))
            reward = reward.item()
            rew_error = np.abs(rew_pred - reward)

            if reward != 0.0:
                print(f"step={step_count}, reward={reward:.2f},"
                      f" reward_pred={rew_pred:.2f}, reward_error={rew_error:.2f},"
                      f" error_threshold={policy.predictor.error_threshold.item()}")

            step_count += 1

            if done:
                print("EPISODE FINISHED", flush=True)
                sys.exit()

            time.sleep(0.05)


if __name__ == "__main__":
    play()
