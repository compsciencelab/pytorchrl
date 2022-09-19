#!/usr/bin/env python3

import os
import time
import torch
import numpy as np
import pytorchrl as prl
from pytorchrl.agent.env import VecEnv
from pytorchrl.envs.atari import atari_train_env_factory
from pytorchrl.agent.actors import OnPolicyActor
from code_examples.train_atari.rnd_ppo.train import get_args


def enjoy():

    args = get_args()

    args.demos_dir = "/tmp/demos_agent"
    if not os.path.isdir(args.demos_dir):
        os.makedirs(args.demos_dir)

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
    env = env()

    policy = OnPolicyActor.create_factory(
        obs_space, action_space, prl.RND_PPO,
        restart_model={"policy_net": os.path.join(args.log_dir, "model.state_dict")},
        recurrent_net=args.recurrent_net,
    )(device)

    # Define initial Tensors
    obs = env.reset()
    env.render()
    done, episode_reward, step = False, 0, 0
    _, rhs, _ = policy.actor_initial_states(torch.tensor(obs))

    obs_rollouts = [obs[:, -1:, :, :]]
    rews_rollouts = []
    actions_rollouts = []

    # Execute episodes
    while not done:

        env.render()
        obs = torch.Tensor(obs).to(device)
        done = torch.Tensor([done]).to(device)

        with torch.no_grad():
            _, clipped_action, _, rhs, _, _ = policy.get_action(obs, rhs, done, deterministic=False)

        # time.sleep(0.05)

        obs, reward, done, info = env.step(clipped_action)
        episode_reward += reward

        obs_rollouts.append(obs[:, -1:, :, :].cpu().numpy())
        rews_rollouts.append(reward.cpu().numpy())
        actions_rollouts.append(clipped_action.cpu().numpy())

        if done:

            print("EPISODE: reward: {}".format(episode_reward.item()), flush=True)

            obs_rollouts.pop(-1)

            num = 0
            filename = os.path.join(
                args.demos_dir, "human_demo_{}".format(num + 1))
            while os.path.exists(filename + ".npz"):
                num += 1
                filename = os.path.join(
                    args.demos_dir, "human_demo_{}".format(num + 1))

            np.savez(
                filename,
                Observation=np.array(np.stack(obs_rollouts).astype(np.uint8)).squeeze(1),
                Reward=np.array(np.stack(rews_rollouts).astype(np.float16)).squeeze(1),
                Action=np.expand_dims(np.array(np.stack(actions_rollouts).astype(np.int8)), axis=1),
                FrameSkip=args.frame_skip,
            )

            print("Saved demo as {}\n".format(filename))

            done, episode_reward = 0, False
            obs = env.reset()


if __name__ == "__main__":
    enjoy()