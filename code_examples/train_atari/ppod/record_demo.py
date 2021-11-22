import os
import sys
import time

import torch
import threading
import numpy as np
from pynput import keyboard
from pytorchrl.agent.env import VecEnv
from pytorchrl.envs.atari import atari_train_env_factory
from code_examples.train_atari.ppo.train import get_args


pressed_keys = set([])


# ACTIONS:

# 'NOOP',  # --> 0
# 'FIRE',  # Space  --> 1
# 'UP',  # W --> 2
# 'RIGHT',  # D --> 3
# 'LEFT',  # A --> 4
# 'DOWN',  # S --> 5
# 'UPRIGHT',  # W + D --> 6
# 'UPLEFT',  # W + A --> 7
# 'DOWNRIGHT',  # S + D --> 8
# 'DOWNLEFT',  # S + A --> 9
# 'UPFIRE',  # Space + W  --> 10
# 'RIGHTFIRE',  # Space + D  --> 11
# 'LEFTFIRE',  # Space + A --> 12
# 'DOWNFIRE',  # Space + S  --> 13
# 'UPRIGHTFIRE',  # Space + W + D  --> 14
# 'UPLEFTFIRE',  # Space + W + A --> 15
# 'DOWNRIGHTFIRE',  # Space + S + D --> 16
# 'DOWNLEFTFIRE',  # Space + S + A --> 17

def create_action():
    action = 0

    if "s" in pressed_keys and "space" in pressed_keys and "a" in pressed_keys:
        print("PRESSED SPACE, A AND S, ACTION 17")
        action = 17
    elif "s" in pressed_keys and "space" in pressed_keys and "a" in pressed_keys:
        print("PRESSED SPACE, D AND S, ACTION 16")
        action = 16
    elif "w" in pressed_keys and "space" in pressed_keys and "a" in pressed_keys:
        print("PRESSED SPACE, W AND D, ACTION 15")
        action = 15
    elif "w" in pressed_keys and "space" in pressed_keys and "d" in pressed_keys:
        print("PRESSED SPACE, W AND D, ACTION 14")
        action = 14
    elif "w" in pressed_keys and "d" in pressed_keys:
        print("PRESSED W AND D, ACTION 6")
        action = 6
    elif "w" in pressed_keys and "a" in pressed_keys:
        print("PRESSED W AND A, ACTION 7")
        action = 7
    elif "s" in pressed_keys and "d" in pressed_keys:
        print("PRESSED S AND D, ACTION 8")
        action = 8
    elif "s" in pressed_keys and "a" in pressed_keys:
        print("PRESSED S AND A, ACTION 9")
        action = 9
    elif "space" in pressed_keys and "w" in pressed_keys:
        print("PRESSED SPACE AND W, ACTION 10")
        action = 10
    elif "space" in pressed_keys and "d" in pressed_keys:
        print("PRESSED SPACE AND D, ACTION 11")
        action = 11
    elif "space" in pressed_keys and "a" in pressed_keys:
        print("PRESSED SPACE AND A, ACTION 12")
        action = 12
    elif "space" in pressed_keys and "s" in pressed_keys:
        print("PRESSED SPACE AND S, ACTION 13")
        action = 13
    elif "w" in pressed_keys:
        print("PRESSED W, ACTION 2")
        action = 2
    elif "s" in pressed_keys:
        print("PRESSED S, ACTION 5")
        action = 5
    elif "d" in pressed_keys:
        print("PRESSED D, ACTION 3")
        action = 3
    elif "a" in pressed_keys:
        print("PRESSED A, ACTION 4")
        action = 4
    elif "space" in pressed_keys:
        print("PRESSED SPACE, ACTION 1")
        action = 1
    return action


def record():
    args = get_args()

    # if not os.path.isdir(args.demos_dir):
    #     os.makedirs(args.demos_dir)

    args.env_id = "MontezumaRevengeNoFrameskip-v4"

    # Define Single Env
    env, action_space, obs_space = VecEnv.create_factory(
        env_fn=atari_train_env_factory,
        env_kwargs={
            "env_id": args.env_id,
            "frame_stack": args.frame_stack,
        },
        vec_env_size=1)

    # Start recording
    env = env()
    obs = env.reset()
    env.render()
    step = 0
    done = False
    episode_reward = 0

    obs_rollouts = [obs]
    rews_rollouts = []
    actions_rollouts = []

    with keyboard.Listener(
            on_press=lambda key: pressed_keys.add(key.char if hasattr(key, "char") else "space"),
            on_release=lambda key: pressed_keys.remove(key.char if hasattr(key, "char") else "space")
    ) as listener:

        threading.Thread(target=listener.join)

        while not done:

            env.render()
            action = create_action()

            obs, reward, done, info = env.step(
                torch.tensor([action]).unsqueeze(1))

            obs_rollouts.append(obs)
            rews_rollouts.append(reward)
            actions_rollouts.append(action)

            step += 1
            episode_reward += reward

            time.sleep(0.05)

            if done:

                print("EPISODE FINISHED: {} steps: ".format(step), flush=True)

                # obs_rollouts.pop(-1)
                #
                # num = 0
                # filename = os.path.join(
                #     args.demos_dir, "human_demo_{}".format(num + 1))
                # while os.path.exists(filename + ".npz"):
                #     num += 1
                #     filename = os.path.join(
                #         args.demos_dir, "human_demo_{}".format(num + 1))
                #
                # for action in actions_rollouts:
                #     assert action in expand.keys()
                #
                # np.savez(
                #     filename,
                #     Observation=np.array(np.stack(obs_rollouts).astype(np.float32)).squeeze(1),
                #     Reward=np.array(np.stack(rews_rollouts).astype(np.float32)).squeeze(1),
                #     Action=np.expand_dims(np.array(np.stack(actions_rollouts).astype(np.float32)), axis=1),
                #     FrameSkip=args.frame_skip,
                # )
                #
                # sys.exit()


if __name__ == "__main__":
    record()
