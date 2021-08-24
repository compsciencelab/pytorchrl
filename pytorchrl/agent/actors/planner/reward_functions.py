
""" 
Reward functions created based on: https://arxiv.org/pdf/1907.02057.pdf
"""
import numpy as np
import torch

def get_reward_function(name):
    if name == "CartPole-v0" or name == "CartPole-v1":
        return cartpole_reward
    elif name == "Acrobot-v1":
        return acrobot_reward
    else:
        raise ValueError("Currently only supports the environments CartPole-v0, CartPole-v1 and Acrobot-v1!")

def cartpole_reward(states, actions):
    x, x_dot, theta, theta_dot = torch.chunk(states, 4, dim=1)

    rewards = np.cos(theta) - 0.01 * x ** 2  # check if its cos already!
    return rewards

def acrobot_reward(states, actions):
    
    cos_theta1 = states[:, 0]
    sin_theta1 = states[:, 1]
    cos_theta2 = states[:, 2]
    sin_theta2 = states[:, 3]
    
    # -cos(a) - cos(a+b) with cos(a+b) = cos(a)*cos(b)-sin(a)*sin(b)
    rewards = - cos_theta1 - (cos_theta1 * cos_theta2 - sin_theta1 * sin_theta2)
    return rewards