from pytorchrl.envs.minigrid.custom_environments.deceiving_rewards import DeceivingRewardsEnv


from gymnasium.envs.registration import register

from minigrid import minigrid_env, wrappers
from minigrid.core import roomgrid
from minigrid.core.world_object import Wall


def register_custom_minigrid_envs():

    # DeceivingRewards
    # ----------------

    register(
        id="MiniGrid-DeceivingRewards-v0",
        entry_point="pytorchrl.envs.minigrid.custom_environments:DeceivingRewardsEnv",
    )


