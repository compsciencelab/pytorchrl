from pytorchrl.agent.actors.reward_functions.gym_reward_functions import pendulum_reward_function


def get_reward_function(env_id):
    if env_id == "Pendulum-v0" or env_id == "Pendulum-v0":
        return pendulum_reward_function