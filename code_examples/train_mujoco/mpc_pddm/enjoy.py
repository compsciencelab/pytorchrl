#!/usr/bin/env python3

import os
import torch
import argparse
from pytorchrl.envs.pybullet import pybullet_test_env_factory
from pytorchrl.agent.actors import MBActor
from pytorchrl.agent.algorithms import MB_MPC
from pytorchrl.utils import LoadFromFile
from code_examples.train_mujoco.mpc_pddm.train import get_args


def enjoy():
    args = get_args()

    # Define single copy of the environment
    env = pybullet_test_env_factory(env_id=args.env_id)
    env.render()

    # Define agent device and agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dynamics_model = MBActor.create_factory(
        args.env_id,
        env.observation_space,
        env.action_space,
        hidden_size=args.hidden_size,
        batch_size=args.mini_batch_size,
        learn_reward_function=args.learn_reward_function,
        checkpoint=os.path.join(args.log_dir, "model.state_dict"))(device)

    algo_factory, algo = MB_MPC.create_factory(args)

    mpc = algo_factory(device=device,
                       actor=dynamics_model,
                       envs=env)

    # Define initial Tensors
    obs, done = env.reset(), False
    _, rhs, _ = dynamics_model.actor_initial_states(torch.tensor(obs))
    episode_reward = 0

    # Execute episodes
    while not done:

        env.render()
        obs = torch.Tensor(obs).view(1, -1).to(device)
        done = torch.Tensor([done]).view(1, -1).to(device)
        with torch.no_grad():
            _, clipped_action, rhs, _ = mpc.acting_step(obs, rhs, done, deterministic=True)
        obs, reward, done, info = env.step(clipped_action.squeeze().cpu().numpy())
        episode_reward += reward

        if done:
            print("EPISODE: reward: {}".format(episode_reward), flush=True)
            done, episode_reward = 0, False
            obs = env.reset()


if __name__ == "__main__":
    enjoy()
