import torch


def halfcheetah_bullet(state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    """   
    HalfCheetahBulletEnv-v0 velocity is 3 idx:
    https://github.com/bulletphysics/bullet3/blob/478da7469a34074aa051e8720734287ca371fd3e/examples/pybullet/gym/pybullet_envs/robot_locomotors.py#L64
    """
    x_velocities = state[:, 3]
    action_penalty = - 0.1 * (torch.sum(action**2, axis=1))
    reward = (x_velocities + action_penalty)[:, None]
    assert reward.shape == (state.shape[0], 1)
    return reward