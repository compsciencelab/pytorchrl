import os
import sys
import time

import torch
import threading
import numpy as np
from pynput import keyboard
from pytorchrl.agent.env import VecEnv
from pytorchrl.envs.obstacle_tower.utils import action_lookup_6, action_lookup_7, action_lookup_8
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

    return tuple(action)


def record():
    args = get_args()

    if args.num_actions == 6:
        action_lookup = action_lookup_6
    elif args.num_actions == 7:
        action_lookup = action_lookup_7
    elif args.num_actions == 8:
        action_lookup = action_lookup_8

    if not os.path.isdir(args.demos_dir):
        os.makedirs(args.demos_dir)

    # Define Single Env
    env, action_space, obs_space = VecEnv.create_factory(
        env_fn=obstacle_train_env_factory,
        env_kwargs={
            "min_floor": args.min_floor,
            "max_floor": args.max_floor,
            "seed_list": args.seed_list,
            "frame_skip": args.frame_skip,
            "reward_shape": False,
            "realtime": True,
            "num_actions": args.num_actions,
            "reduced_actions": args.reduced_action_space,
        },
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

            if args.num_actions == 6 and tuple(action) not in action_lookup_6.keys():
                action = (1, 0, 0, 0)
            elif args.num_actions == 7 and tuple(action) not in action_lookup_7.keys():
                 action = (0, 0, 0, 0)
            elif args.num_actions == 8 and tuple(action) not in action_lookup_8.keys():
                 action = (0, 0, 0, 0)

            print("tuple action:", action)
            print("int action:", action)
            print()

            obs, reward, done, info = env.step(torch.tensor(
                [action_lookup[action]]).unsqueeze(0))

            obs_rollouts.append(obs)
            rews_rollouts.append(reward)
            actions_rollouts.append(action_lookup[action])

            step += 1
            episode_reward += reward

            time.sleep(0.05)

            if done:

                print("EPISODE FINISHED: {} steps: ".format(step), flush=True)

                obs_rollouts.pop(-1)

                num = 0
                filename = os.path.join(
                    args.demos_dir, "reward_demo_{}".format(num + 1))
                while os.path.exists(filename + ".npz"):
                    num += 1
                    filename = os.path.join(
                        args.demos_dir, "reward_demo_{}".format(num + 1))

                np.savez(
                    filename,
                    Observation=np.array(np.stack(obs_rollouts).astype(np.uint8)).squeeze(1),
                    Reward=np.array(np.stack(rews_rollouts).astype(np.float16)).squeeze(1),
                    Action=np.expand_dims(np.array(np.stack(actions_rollouts).astype(np.int8)), axis=1),
                    FrameSkip=args.frame_skip,
                )

                sys.exit()


if __name__ == "__main__":
    record()
