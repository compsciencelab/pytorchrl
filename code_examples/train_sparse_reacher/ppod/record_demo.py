import os
import sys
import time

import gym
import torch
import argparse
import threading
import pybulletgym
import numpy as np
from pynput import keyboard
from pytorchrl.agent.env import VecEnv
from pytorchrl.envs.common import FrameStack, FrameSkip

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

gym.envs.register(
    id='SparseReacher-v1',
    entry_point='pytorchrl.envs.pybullet.sparse_reacher:SparseReacherBulletEnv',
    max_episode_steps=150,
    reward_threshold=18.0,
)

def sparse_reacher_env_factory(seed=0, index_worker=0, index_env=0, frame_skip=0, frame_stack=1):
    """ Create sparse sparse_reacher environment instances."""

    env = gym.make("SparseReacher-v1")
    env.seed(seed + index_worker + index_env)

    if frame_skip > 0:
        env = FrameSkip(env, skip=frame_skip)

    if frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    env.render()

    return env


pressed_keys = set([])

# #{0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [1, 0], 4: [1, 1], 5: [1, 2], 6: [2, 0], 7: [2, 1], 8: [2, 2]}
expand = {0: [0, 0], 1: [0, 1], 2: [0, 2], 3: [1, 0], 4: [1, 1], 5: [1, 2], 6: [2, 0], 7: [2, 1], 8: [2, 2]}
reduce = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 3, (1, 1): 4, (1, 2): 5, (2, 0): 6, (2, 1): 7, (2, 2): 8}


def create_action():
    action = -1
    while action == -1:
        if "w" in pressed_keys:
            action = [0, 0]
        elif "s" in pressed_keys:
            action = [0, 1]
        if "d" in pressed_keys:
            action = [1, 0]
        elif "a" in pressed_keys:
            action = [1, 1]
    return action


def record():
    args = get_args()

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # Define Single Env
    train_envs_factory, action_space, obs_space = VecEnv.create_factory(
        vec_env_size=1, log_dir=None,
        env_fn=sparse_reacher_env_factory, env_kwargs={
            # "seed": args.seed,
            "frame_skip": args.frame_skip,
            "frame_stack": args.frame_stack})

    env = train_envs_factory(args.device)

    # Start recording
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
            obs, reward, done, info = env.step(torch.tensor(action).unsqueeze(1))

            obs_rollouts.append(obs)
            rews_rollouts.append(reward)
            actions_rollouts.append(reduce[tuple(action)])

            step += 1
            episode_reward += reward

            time.sleep(1.0)

            if done:

                import ipdb; ipdb.set_trace()
                print("EPISODE FINISHED: {} steps: ".format(step), flush=True)

                obs_rollouts.pop(-1)

                num = 0

                filename = os.path.join(
                    args.save_dir, "sparse_reacher_demo_{}".format(num))
                while os.path.exists(filename):
                    num += 1
                    filename = os.path.join(
                        args.save_dir, "sparse_reacher_demo_{}".format(num))

                for action in actions_rollouts:
                    assert action in expand.keys()

                np.savez(
                    filename,
                    observations=np.array(np.stack(obs_rollouts).astype(np.float32)).squeeze(1),
                    rewards=np.array(np.stack(rews_rollouts).astype(np.float32)).squeeze(1),
                    actions=np.expand_dims(np.array(np.stack(actions_rollouts).astype(np.float32)), axis=1)
                )

                sys.exit()


def get_args():
    dirname = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='demos')
    parser.add_argument('--device', default="cpu", help='compute device')
    parser.add_argument('--frame-skip', type=int, default=0, help='Number of frame to skip for each action (default no skip)')
    parser.add_argument('--frame-stack', type=int, default=0, help='Number of frame to stack in observation (default no stack)')
    parser.add_argument('--save-dir', default=os.path.join(dirname, 'demos'), help='path to target dir')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    record()
