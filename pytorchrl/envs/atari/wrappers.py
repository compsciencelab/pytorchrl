"""wrappers from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py"""

import os

os.environ.setdefault('PATH', '')

import gym
import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)
from pytorchrl.envs.common import FrameStack
from pytorchrl.envs.atari.utils import imdownscale


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action

        self.last_action = action
        return self.env.step(action)

    def reset(self):
        self.last_action = 0
        return self.env.reset()


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        self.total_reward = 0.0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.total_reward += reward
        info['EpisodicReward'] = self.total_reward

        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        info['Lives'] = lives
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            self.total_reward = 0.0
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class ClipRewardEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_reward = 0.0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.total_reward += reward
        info['UnclippedReward'] = self.total_reward

        reward = np.sign(reward)
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.total_reward = 0.0
        return self.env.reset(**kwargs)


class ScaleRewardEnv(gym.Wrapper):
    def __init__(self, env, scaling=0.001):
        gym.Wrapper.__init__(self, env)
        self.scaling = scaling
        self.total_reward = 0.0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_reward += reward
        info['UnscaledReward'] = self.total_reward
        reward = self.scaling * reward
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.total_reward = 0.0
        return self.env.reset(**kwargs)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class MontezumaVisitedRoomEnv(gym.Wrapper):
    def __init__(self, env, room_address):
        gym.Wrapper.__init__(self, env)
        self.room_address = room_address
        self.visited_rooms = set()  # Only stores unique numbers.

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        ram = self.unwrapped.ale.getRAM()
        assert len(ram) == 128
        self.visited_rooms.add(ram[self.room_address])
        info['VisitedRooms'] = len(self.visited_rooms)
        return state, reward, done, info

    def reset(self):
        self.visited_rooms.clear()
        return self.env.reset()


class MontezumaEmbeddingsEnv(gym.Wrapper):
    def __init__(self, env, use_domain_knowledge=False):
        gym.Wrapper.__init__(self, env)

        # pos 0: The current frame of the episode.
        # pos 3: The current screen. Room?
        # pod 57: room level?
        # pos 19, 20, 21: The score, represented in Binary Coded Decimal. This is, every nibble represents a decimal digit
        # pos 42: joe_x, from 0 to 153
        # pos 43: joe_y, from 0 to 122
        # pos 52: agent orientation, 76 or 128
        # pos 65: inventory
        #       mallet +1
        #       key +2
        #       key +4
        #       key +8
        #       key +16
        #       sword +32
        #       sword +64
        #       torch +128

        self.room_address = 3
        self.room_level = 57
        self.joe_x = 42
        self.joe_y = 43
        self.joe_inventory = 65
        self.use_domain_knowledge = use_domain_knowledge

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if self.use_domain_knowledge:
            ram = self.unwrapped.ale.getRAM()
            assert len(ram) == 128
            embed_state = np.array(
                [
                    ram[self.joe_x],
                    ram[self.joe_y],
                    ram[self.room_address] + 24 * ram[self.room_level],
                    ram[self.joe_inventory],
                ]
            )
            print(embed_state)
        else:
            embed_state = imdownscale(state[:, :, -1])

        info.update({"StateEmbeddings": embed_state})

        return state, reward, done, info


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=1, scale=False):
    """Configure environment for DeepMind-style Atari"""
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = FrameStack(env, frame_stack)
    return env


def make_atari(env_id, max_episode_steps=None, sticky_actions=False):
    env = gym.make(env_id)
    env._max_episode_steps = max_episode_steps * 4
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if sticky_actions:
        env = StickyActionEnv(env)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
