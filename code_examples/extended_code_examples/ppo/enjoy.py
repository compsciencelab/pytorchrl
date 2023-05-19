#!/usr/bin/env python3

import os
import time
import torch
import argparse
import pytorchrl as prl
from pytorchrl.envs.atari import atari_test_env_factory
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor
from pytorchrl.utils import LoadFromFile
from pytorchrl.agent.env.env_wrappers import TransposeImagesIfRequired
from code_examples.train_atari.ppo.train import get_args


def enjoy():

    args = get_args()

    # Define single copy of the environment
    env = atari_test_env_factory(env_id=args.env_id, frame_stack=args.frame_stack)
    env = TransposeImagesIfRequired(env)
    env.render()

    # Define agent device and agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    policy = OnPolicyActor.create_factory(
        env.observation_space, env.action_space, prl.PPO,
        restart_model=os.path.join(args.log_dir, "model.state_dict"))(device)

    # Define initial Tensors
    obs, done = env.reset(), False
    _, rhs, _ = policy.actor_initial_states(torch.tensor(obs))
    episode_reward = 0

    # Execute episodes
    while not done:

        env.render()
        time.sleep(0.01)
        obs = torch.Tensor(obs).unsqueeze(0).to(device)
        done = torch.Tensor([done]).unsqueeze(0).to(device)
        with torch.no_grad():
            _, clipped_action, _, rhs, _, _ = policy.get_action(obs, rhs, done, deterministic=False)
        obs, reward, done, info = env.step(clipped_action.squeeze().cpu().numpy())
        episode_reward += reward

        if done:
            print("EPISODE: reward: {}".format(episode_reward), flush=True)
            done, episode_reward = 0, False
            obs = env.reset()


if __name__ == "__main__":
    enjoy()
