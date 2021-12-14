import torch


def halfcheetah_reward_function(state: torch.Tensor, action: torch.Tensor)-> torch.Tensor:
    """Half Cheetah is a 2D robot with 7 rigid links, including 2 legs and a torso. There
       are 6 actuators located at 6 joints respectively. The goal is to run forward as fast as possible, while
       keeping control inputs small. The observation include the (angular) position and velocity of all the
       joints (including the root joint, whose position specifies the robot’s position in the world coordinate),
       except for the x position of the root joint. The reward is the x direction velocity plus penalty for
       control inputs. (https://arxiv.org/pdf/1907.02057.pdf)
       
       pybullet halfcheetah: https://github.com/benelot/pybullet-gym/blob/master/pybulletgym/envs/mujoco/robots/locomotors/half_cheetah.py
    """
    # reward =  x_dot_t − 0.1||a_t||2
    
    x_dot = state[:, 8:]
    a_t = (action ** 2).sum(1)[:, None]
    reward = x_dot - 0.1*a_t
    return reward