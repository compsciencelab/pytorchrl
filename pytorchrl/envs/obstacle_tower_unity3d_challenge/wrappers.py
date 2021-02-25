import numpy as np
import gym
from gym import spaces
from pytorchrl.envs.obstacle_tower_unity3d_challenge.utils import (
    box_is_placed, box_location, place_location, reduced_action_lookup_6,
    reduced_action_lookup_7)


class BasicObstacleEnv(gym.Wrapper):
    def __init__(self, env, min_floor, max_floor):

        gym.Wrapper.__init__(self, env)

        self.reached_floor = 0
        self._min_floor = min_floor
        self._max_floor = max_floor
        self.count = 0
        self.last_time = 3000

        self.seed = None
        self.start_floor = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if info['current_floor'] > self.reached_floor:
            self.reached_floor = info['current_floor']

        if info['current_floor'] > self._max_floor:
            done = True

        info['seed'] = self.seed
        info['start'] = self.start_floor
        info['floor'] = self.reached_floor

        num_keys = info["total_keys"]
        self.picked_key = num_keys > self._previous_keys
        self._previous_keys = num_keys
        self.last_time = info['time_remaining']

        return obs, reward, done, info

    def reset(self, **kwargs):

        self._previous_keys = 0
        self.puzzle_solved = False

        self.seed = np.random.randint(0, 100)
        self.env.unwrapped.seed(self.seed)

        self.start_floor = np.random.randint(
            self._min_floor, self.reached_floor if self.reached_floor != 0 else 1)
        self.env.unwrapped.floor(self.start_floor)
        self.reached_floor = 0

        config = {"total-floors": self._max_floor + 2}
        self.count += 1

        return self.env.reset(config=config, **kwargs)

class BasicObstacleEnvTest(gym.Wrapper):
    def __init__(self, env, min_floor, max_floor, seed_list=[1001, 1002, 1003, 1004, 1005]):

        gym.Wrapper.__init__(self, env)

        self.reached_floor = None
        self._min_floor = min_floor
        self._max_floor = max_floor
        self.last_time = 3000

        self.seed_index = -1
        self.seed_list = seed_list
        self.reached_floors = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if info['current_floor'] > self.reached_floor:
            self.reached_floor = info['current_floor']

        info['seed'] = self.seed
        info['start'] = self.start_floor
        info['floor'] = self.reached_floor

        num_keys = info["total_keys"]
        self.picked_key = num_keys > self._previous_keys
        self._previous_keys = num_keys
        self.last_time = info['time_remaining']

        return obs, reward, done, info

    def reset(self, **kwargs):

        if self.reached_floor is not None:
            self.reached_floors.append(self.reached_floor)
            print("Seed {}, reahed foor {}".format(self.seed, self.reached_floor))
        if self.seed_index == len(self.seed_list) - 1:
            print("Average of all seeds {}".format(np.mean(self.reached_floors)))
            self.reached_floors = []

        self._previous_keys = 0
        self.seed_index = (self.seed_index + 1) % len(self.seed_list)
        self.seed = self.seed_list[self.seed_index]
        self.env.unwrapped.seed(self.seed)

        self.start_floor = self._min_floor
        self.env.unwrapped.floor(self.start_floor)
        self.reached_floor = 0
        config = {"total-floors": self._max_floor + 2}

        return self.env.reset(config=config, **kwargs)

class RewardShapeObstacleEnv(gym.Wrapper):

    def __init__(self, env, killed_reward=2):
        gym.Wrapper.__init__(self, env)
        self._killed_reward = killed_reward
        self.time_remaining = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Picked key
        if self.env.picked_key:
            reward += 1.01

        found_box, _ = box_location(obs)
        if found_box:
            reward += 0.002

        found_place, _ = place_location(obs)
        if found_place:
            reward += 0.001

        # Account for solving puzzle
        if box_is_placed(obs) and (not self.puzzle_solved) and (reward > 0.08):
            self.puzzle_solved = True
            reward += 1.5

        if self.time_remaining == None:
            self.time_remaining = info['time_remaining']
        if info['time_remaining'] > self.time_remaining:
            reward += 0.002

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.puzzle_solved = False
        self.time_remaining = None
        return self.env.reset(**kwargs)

class ReducedActionEnv(gym.Wrapper):
    def __init__(self, env, num_actions=6):

        if num_actions == 6:
            _action_lookup = reduced_action_lookup_6
        elif num_actions == 7:
            _action_lookup = reduced_action_lookup_7
        else:
            ValueError("No lookup table for num reduced actions {}".format(
                num_actions))

        env.unwrapped._flattener.action_lookup = _action_lookup
        num_actions = len(env.unwrapped._flattener.action_lookup)
        env.unwrapped._flattener.action_space = spaces.Discrete(num_actions)
        env.unwrapped._action_space = env.unwrapped._flattener.action_space
        gym.Wrapper.__init__(self, env)