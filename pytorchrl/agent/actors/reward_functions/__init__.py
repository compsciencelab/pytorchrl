from pytorchrl.agent.actors.reward_functions.gym_reward_functions import pendulum, halfcheetah_mujoco
from pytorchrl.agent.actors.reward_functions.pybullet_reward_functions import halfcheetah_bullet
"""Good Resource of reward functions: https://arxiv.org/pdf/1907.02057.pdf 
    including gym envrionemnts as well as pybullet environments
"""

def get_reward_function(env_id):
    # gym env
    if env_id == "Pendulum-v0" or env_id == "Pendulum-v1":
        return pendulum
    if env_id == "HalfCheetah-v2" or env_id == "HalfCheetah-v3":
        return halfcheetah_mujoco
    
    # pybullet env
    elif env_id == "HalfCheetahBulletEnv-v0":
        return halfcheetah_bullet
    