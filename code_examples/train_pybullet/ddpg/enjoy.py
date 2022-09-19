#!/usr/bin/env python3

import os
import torch
import argparse
from pytorchrl.envs.pybullet import pybullet_test_env_factory
from pytorchrl.agent.actors import OffPolicyActor, get_feature_extractor
from pytorchrl.utils import LoadFromFile
from code_examples.train_pybullet.ddpg.train import get_args


def enjoy():

    args = get_args()

    # Define single copy of the environment
    env = pybullet_test_env_factory(env_id=args.env_id)
    env.render()

    # Define agent device and agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    policy = OffPolicyActor.create_factory(
        env.observation_space, env.action_space,
        restart_model=os.path.join(args.log_dir, "model.state_dict"))(device)

    # Define initial Tensors
    obs, done = env.reset(), False
    _, rhs, _ = policy.actor_initial_states(torch.tensor(obs))
    episode_reward = 0

    # Execute episodes
    while not done:

        env.render()
        obs = torch.Tensor(obs).view(1, -1).to(device)
        done = torch.Tensor([done]).view(1, -1).to(device)
        with torch.no_grad():
            _, clipped_action, _, rhs, _ = policy.get_action(obs, rhs, done, deterministic=True)
        obs, reward, done, info = env.step(clipped_action.squeeze().cpu().numpy())
        episode_reward += reward

        if done:
            print("EPISODE: reward: {}".format(episode_reward), flush=True)
            done, episode_reward = 0, False
            obs = env.reset()


if __name__ == "__main__":
    enjoy()
