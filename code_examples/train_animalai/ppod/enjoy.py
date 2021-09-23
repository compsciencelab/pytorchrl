#!/usr/bin/env python3

import os
import torch
import pytorchrl as prl
from pytorchrl.agent.env import VecEnv
from pytorchrl.agent.actors import OnPolicyActor
from pytorchrl.envs.animal_olympics.animal_olympics_env_factory import animal_train_env_factory
from code_examples.train_animalai.ppod.train import get_args


def enjoy():

    args = get_args()

    # Define single copy of the environment
    arena_file = os.path.dirname(os.path.abspath(__file__)) + "/arenas/"
    env, action_space, obs_space = VecEnv.create_factory(
        env_fn=animal_train_env_factory,
        env_kwargs={
            "arenas_dir": arena_file,
            "frame_skip": args.frame_skip,
            "frame_stack": args.frame_stack,
            "inference": True,
        }, vec_env_size=1)

    # Define agent device and policy
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = OnPolicyActor.create_factory(
        obs_space, action_space, prl.PPO,
        restart_model=args.restart_model,
        recurrent_nets=args.recurrent_nets)(device)

    # Define initial Tensors
    env = env()
    obs = env.reset()
    done, episode_reward, step = False, 0, 0
    _, rhs, _ = policy.actor_initial_states(torch.tensor(obs))

    # Execute episodes
    while not done:

        obs = torch.Tensor(obs).to(device)
        done = torch.Tensor([done]).to(device)

        with torch.no_grad():
            _, clipped_action, _, rhs, _, _ = policy.get_action(obs, rhs, done, deterministic=True)

        obs, reward, done, info = env.step(clipped_action.squeeze().cpu().numpy())
        episode_reward += reward

        if done:
            print("EPISODE: reward: {}".format(episode_reward.item()), flush=True)
            done, episode_reward = 0, False
            obs = env.reset()


if __name__ == "__main__":
    enjoy()
