#!/usr/bin/env python3

import os
import time
import torch
import pytorchrl as prl
from pytorchrl.agent.env import VecEnv
from pytorchrl.envs.atari import atari_train_env_factory
from pytorchrl.agent.actors import OnPolicyActor
from code_examples.train_atari.ppod.train import get_args


def enjoy():

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

    policy = OnPolicyActor.create_factory(
        obs_space, action_space, prl.PPO,
        restart_model=os.path.join(args.log_dir, "model.state_dict"),
        recurrent_net=False,
    )(device)

    # Define initial Tensors
    env = env()
    obs = env.reset()
    env.render()
    done, episode_reward, step = False, 0, 0

    _, rhs, _ = policy.actor_initial_states(torch.tensor(obs))

    # Execute episodes
    while not done:

        env.render()
        obs = torch.Tensor(obs).to(device)
        done = torch.Tensor([done]).to(device)

        with torch.no_grad():
            _, clipped_action, _, rhs, _, _ = policy.get_action(obs, rhs, done, deterministic=True)

        print(clipped_action)
        time.sleep(0.01)

        obs, reward, done, info = env.step(clipped_action)
        episode_reward += reward

        if done:
            print("EPISODE: reward: {}".format(episode_reward.item()), flush=True)
            done, episode_reward = 0, False
            obs = env.reset()


if __name__ == "__main__":
    enjoy()
