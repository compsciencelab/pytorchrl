import torch
import math


def pendulum_reward_function(state: torch.Tensor, action: torch.Tensor)-> torch.Tensor:
    max_torque = 2.0
    th = torch.arccos(state[:, 0][:, None])
    thdot = state[:, 2][:, None]
    action = torch.clamp(action, -max_torque, max_torque)

    reward = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (action ** 2)
    return -reward

def angle_normalize(x: torch.Tensor):
    return ((x + math.pi) % (2 * math.pi)) - math.pi

