#!/usr/bin/env python3

import os
import torch
import pytorchrl as prl
from pytorchrl.agent.env import VecEnv
from pytorchrl.envs.obstacle_tower.obstacle_tower_env_factory import obstacle_train_env_factory
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor
from code_examples.train_obstacle_tower.ppod.train import get_args


def enjoy():

    args = get_args()

    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define single copy of the environment
    env, action_space, obs_space = VecEnv.create_factory(
        env_fn=obstacle_train_env_factory,
        env_kwargs={
            "min_floor": args.min_floor,
            "max_floor": args.max_floor,
            "seed_list": args.seed_list,
            "frame_skip": args.frame_skip,
            "frame_stack": args.frame_stack,
            "reward_shape": args.reward_shape,
            "reduced_actions": args.reduced_action_space,
            "num_actions": args.num_actions,
            "realtime": True,
        }, vec_env_size=1)(device)

    policy = OnPolicyActor.create_factory(
        obs_space, action_space, prl.PPO,
        feature_extractor_network=get_feature_extractor(args.nn),
        recurrent_nets=args.recurrent_nets,
#        restart_model=os.path.join(args.log_dir, "model.state_dict")
    )(device)

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

        obs, reward, done, info = env.step(clipped_action)
        episode_reward += reward

        if done:
            print("EPISODE: reward: {}".format(episode_reward.item()), flush=True)
            done, episode_reward = 0, False
            obs = env.reset()


if __name__ == "__main__":
    enjoy()
