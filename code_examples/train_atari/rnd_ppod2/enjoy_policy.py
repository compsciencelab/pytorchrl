#!/usr/bin/env python3

import os
import time
import torch
import numpy as np
import pytorchrl as prl
from pytorchrl.agent.env import VecEnv
from pytorchrl.agent.storages import PPOD2RebelBuffer
from pytorchrl.envs.atari import atari_train_env_factory
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor
from code_examples.train_atari.rnd_ppod2.train import get_args


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
    env = env()

    policy = OnPolicyActor.create_factory(
        obs_space, action_space, prl.RND_PPO,
        restart_model={"policy_net": args.restart_reference_model},
        recurrent_net=args.recurrent_net,
    )(device)

    storage_factory = PPOD2RebelBuffer.create_factory(
        size=args.num_steps, gae_lambda=args.gae_lambda,
        reward_predictor_factory=get_feature_extractor(args.feature_extractor_net),
        reward_predictor_net_kwargs={
            "input_space": env.observation_space,
            "output_sizes": [256, 448, 1],
            "final_activation": False,
        },
        restart_reward_predictor_net=args.restart_reference_model,
        target_reward_demos_dir=os.path.join(args.log_dir, "reward_demos"),
        initial_reward_threshold=1.0,
    )(device, policy, None, env)
    policy.try_load_from_checkpoint()

    # Define initial Tensors
    obs = env.reset()
    env.render()
    done, episode_reward, step = False, 0, 0

    _, rhs, _ = policy.actor_initial_states(torch.tensor(obs))

    # Execute episodes
    step_count = 0
    while not done:

        env.render()
        obs = torch.Tensor(obs).to(device)
        done = torch.Tensor([done]).to(device)

        with torch.no_grad():
            _, clipped_action, _, rhs, _, _ = policy.get_action(obs, rhs, done, deterministic=True)
            rew_pred = policy.predictor.reward_predictor(obs).item()

        obs, reward, done, info = env.step(clipped_action)
        episode_reward += reward
        rew_error = np.abs(rew_pred - reward.item())

        if reward != 0.0:
            print(f"step={step_count}, reward={reward.item():.2f},"
                  f" reward_pred={rew_pred:.2f}, reward_error={rew_error:.2f}")

        if done:
            print("EPISODE: reward: {}".format(episode_reward.item()), flush=True)
            done, episode_reward = 0, False
            obs = env.reset()

        time.sleep(0.05)


if __name__ == "__main__":
    enjoy()
