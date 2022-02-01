import torch
import math

"""Good Resource of reward functions: https://arxiv.org/pdf/1907.02057.pdf 
    including gym envrionemnts as well as pybullet environments
"""


def pendulum(state: torch.Tensor, action: torch.Tensor)-> torch.Tensor:
    max_torque = 2.0
    th = torch.arccos(state[:, 0][:, None])
    thdot = state[:, 2][:, None]
    action = torch.clamp(action, -max_torque, max_torque)

    reward = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (action ** 2)
    return -reward

def angle_normalize(x: torch.Tensor):
    return ((x + math.pi) % (2 * math.pi)) - math.pi


def halfcheetah_mujoco(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    """
    First 8 values in the state are position data
    other 9 are position velocities (x,y,z) and rest angular
    -> idx 8 is x_velocitiy
    """
    x_velocities = state[:, 8]
    action_penalty = - 0.1 * (torch.sum(action**2, axis=1))
    reward = (x_velocities + action_penalty)[:, None]
    assert reward.shape == (state.shape[0], 1)
    return reward
