import torch
import math

"""Good Resource of reward functions: https://arxiv.org/pdf/1907.02057.pdf 
    including gym envrionemnts as well as pybullet environments
"""


def cartpole(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    """
    Based on https://arxiv.org/pdf/1907.02057.pdf
    reward = cos(θ_t) - 0.01x²
    """
    x, x_dot, theta, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]

    reward = torch.cos(theta)[:, None] - 0.01 * x[:, None] ** 2
    return reward


def pendulum(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    # Original env reward function seems not to work!
    # max_torque = 2.0
    # th = torch.acos(state[:, 0][:, None])
    # thdot = state[:, 2][:, None]
    # action = torch.clamp(action, -max_torque, max_torque)

    # reward = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (action ** 2)

    # reward function from Paper: https://arxiv.org/pdf/1907.02057.pdf
    cos_theta, sin_theta, theta_dot = state[:, 0], state[:, 1], state[:, 2]
    reward = - cos_theta[:, None] - 0.1 * sin_theta[:, None] - 0.1 * theta_dot[:, None] ** 2 - 0.001 * action ** 2
    return reward


def angle_normalize(x: torch.Tensor):
    return ((x + math.pi) % (2 * math.pi)) - math.pi


def inverted_pendulum_mujoco(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    """
    Env info: https://github.com/openai/gym/blob/master/gym/envs/mujoco/inverted_pendulum.py
    Reward function based on: https://arxiv.org/pdf/1907.02057.pdf
    
    reward = - theta², where theta = state[1]
    
    """
    reward = - state[:, 1][:, None] ** 2
    return reward


def halfcheetah_mujoco(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    """
    First 8 values in the state are position data
    other 9 are position velocities (x,y,z) and rest angular
    -> idx 8 is x_velocitiy
    """
    x_velocities = state[:, 8]
    action_penalty = - 0.1 * (torch.sum(action ** 2, axis=1))
    reward = (x_velocities + action_penalty)[:, None]
    assert reward.shape == (state.shape[0], 1)
    return reward
