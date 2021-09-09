import os
import sys
import time

import torch
import threading
import numpy as np
from pynput import keyboard
from pytorchrl.agent.env import VecEnv
from pytorchrl.envs.obstacle_tower.utils import action_lookup_8
from pytorchrl.envs.obstacle_tower.obstacle_tower_env_factory import obstacle_train_env_factory
from code_examples.train_obstacle_tower.ppod.train import get_args


pressed_keys = set([])

def create_action():
    action = [0, 0, 0, 0]

    if "w" in pressed_keys:
        print("PRESSED W")
        action[0] = 1
    elif "s" in pressed_keys:
        print("PRESSED S")
        action[0] = 2

    if "a" in pressed_keys:
        print("PRESSED A")
        action[1] = 1
    elif "d" in pressed_keys:
        print("PRESSED D")
        action[1] = 2

    if "space" in pressed_keys:
        print("PRESSED SPACE")
        action[2] = 1

    # if "d" in pressed_keys:
    #     print("PRESSED D")
    #     action[3] = 1
    # elif "a" in pressed_keys:
    #     print("PRESSED A")
    #     action[3] = 2

    return tuple(action)


def record():
    args = get_args()

    if not os.path.isdir(args.demos_dir):
        os.makedirs(args.demos_dir)

    # Define Single Env
    env, action_space, obs_space = VecEnv.create_factory(
        env_fn=obstacle_train_env_factory,
        env_kwargs={"frame_skip": args.frame_skip, "frame_stack": args.frame_stack,
                    "reward_shape": True, "realtime": True},
        vec_env_size=1)

    # Start recording
    env = env()
    obs = env.reset()
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

            action = create_action()

            print(action)

            if tuple(action) not in action_lookup_8.keys():
                action = (0, 0, 0, 0)

            print(action)
            print()

            obs, reward, done, info = env.step(torch.tensor(
                [action_lookup_8[action]]).unsqueeze(0))

            obs_rollouts.append(obs)
            rews_rollouts.append(reward)
            actions_rollouts.append(action_lookup_8[action])

            step += 1
            episode_reward += reward

            # time.sleep(0.01)

            if done:

                print("EPISODE FINISHED: {} steps: ".format(step), flush=True)

                obs_rollouts.pop(-1)

                num = 0
                filename = os.path.join(
                    args.demos_dir, "obstacletower_demo_{}".format(num))
                while os.path.exists(filename + ".npz"):
                    num += 1
                    filename = os.path.join(
                        args.demos_dir, "obstacletower_demo_{}".format(num))

                np.savez(
                    filename,
                    Observation=np.array(np.stack(obs_rollouts).astype(np.float32)).squeeze(1),
                    Reward=np.array(np.stack(rews_rollouts).astype(np.float32)).squeeze(1),
                    Action=np.expand_dims(np.array(np.stack(actions_rollouts).astype(np.float32)), axis=1)
                )

                sys.exit()


if __name__ == "__main__":
    record()
