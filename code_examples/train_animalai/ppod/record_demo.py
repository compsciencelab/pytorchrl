import os
import sys
import time

import torch
import threading
import numpy as np
from pynput import keyboard
from pytorchrl.agent.env import VecEnv
from pytorchrl.envs.animal_olympics.animal_olympics_env_factory import animal_train_env_factory
from code_examples.train_animalai.ppod.train import get_args


pressed_keys = set([])
expand = {0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [1, 0], 4: [1, 1], 5: [1, 2], 6: [2, 0], 7: [2, 1], 8: [2, 2]}
reduce = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 3, (1, 1): 4, (1, 2): 5, (2, 0): 6, (2, 1): 7, (2, 2): 8}


def create_action():
    action = [0, 0]
    if "w" in pressed_keys:
        print("PRESSED W")
        action[0] = 1
    elif "s" in pressed_keys:
        print("PRESSED S")
        action[0] = 2
    if "d" in pressed_keys:
        print("PRESSED D")
        action[1] = 1
    elif "a" in pressed_keys:
        print("PRESSED A")
        action[1] = 2
    return action


def record():
    args = get_args()

    if not os.path.isdir(args.demos_dir):
        os.makedirs(args.demos_dir)

    # Define Single Env
    # 1. Define Train Vector of Envs
    env, action_space, obs_space = VecEnv.create_factory(
        env_fn=animal_train_env_factory,
        env_kwargs={
            "arenas_dir": args.arenas_dir,
            "frame_skip": args.frame_skip,
            "inference": True,
        }, vec_env_size=1)

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
            on_press=lambda key: pressed_keys.add(key.char),
            on_release=lambda key: pressed_keys.remove(key.char)
    ) as listener:

        threading.Thread(target=listener.join)

        while not done:

            action = create_action()
            obs, reward, done, info = env.step(
                torch.tensor([reduce[tuple(action)]]).unsqueeze(1))

            obs_rollouts.append(obs)
            rews_rollouts.append(reward)
            actions_rollouts.append(reduce[tuple(action)])

            step += 1
            episode_reward += reward

            time.sleep(0.01)

            if done:

                print("EPISODE FINISHED: {} steps: ".format(step), flush=True)

                obs_rollouts.pop(-1)

                num = 0
                filename = os.path.join(
                    args.demos_dir, "human_demo_{}".format(num + 1))
                while os.path.exists(filename + ".npz"):
                    num += 1
                    filename = os.path.join(
                        args.demos_dir, "human_demo_{}".format(num + 1))

                for action in actions_rollouts:
                    assert action in expand.keys()

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
