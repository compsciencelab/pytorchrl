import csv
import time
import json
import torch
from copy import copy
import numpy as np
import gym
from gym.spaces.box import Box
from glob import glob
import os.path as osp


class TransposeImagesIfRequired(gym.ObservationWrapper):
    """
    When environment observations are images, this wrapper transposes
    the axis. It is useful when the images have shape (W,H,C), as they can be
    transposed "on the fly" to (C,W,H) for PyTorch convolutions to be applied.

    Parameters
    ----------
    env : gym.Env
        Original Gym environment, previous to applying the wrapper.
    op : list
        New axis ordering.
    """

    def __init__(self, env=None, op=[2, 0, 1]):
        """Transpose observation space for images"""
        super(TransposeImagesIfRequired, self).__init__(env)

        self.op = op

        if isinstance(self.observation_space, gym.spaces.Box) and \
                len(self.observation_space.shape) == 3:
            obs_shape = self.observation_space.shape
            self.observation_space = Box(
                self.observation_space.low[0, 0, 0],
                self.observation_space.high[0, 0, 0],
                [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
                dtype=self.observation_space.dtype)

        elif isinstance(self.observation_space, gym.spaces.Dict):
            for k in self.observation_space.spaces:
                if isinstance(self.observation_space[k], gym.spaces.Box) and \
                        len(self.observation_space[k].shape) == 3:
                    obs_shape = self.observation_space[k].shape
                    self.observation_space[k] = Box(
                        self.observation_space[k].low[0, 0, 0],
                        self.observation_space[k].high[0, 0, 0],
                        [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
                        dtype=self.observation_space.dtype)

    def observation(self, ob):
        """Transpose observation"""

        if isinstance(ob, dict):
            for k in ob:
                if len(ob[k].shape) == 3:
                    ob[k] = ob[k].transpose(self.op[0], self.op[1], self.op[2])
        else:
            if len(ob.shape) == 3:
                ob = ob.transpose(self.op[0], self.op[1], self.op[2])

        return ob


class PyTorchEnv(gym.Wrapper):
    """
    This wrapper turns obs, reward's and done's from numpy arrays to pytorch
    tensors and places them in the specified device, facilitating interaction
    between the environment and the actor critic function approximators (NNs).

    Parameters
    ----------
    env : gym.Env
        Original vector environment, previous to applying the wrapper.
    device : torch.device
        CPU or specific GPU where obs, reward's and done's are placed after
        being transformed into pytorch tensors.

    Attributes
    ----------
    device : torch.device
        CPU or specific GPU where obs, reward's and done's are placed after
        being transformed into pytorch tensors.
    """
    def __init__(self, env, device):
        super(PyTorchEnv, self).__init__(env)
        self.env = env
        self.device = device

    def reset(self):
        """New vec env reset function"""
        obs = self.env.reset()
        if isinstance(obs, dict):
            for k in obs:
                obs[k] = torch.from_numpy(obs[k]).float().to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step(self, action):
        """New vec env step_wait function"""

        if isinstance(action, dict):
            for k in action:
                if isinstance(action[k], torch.Tensor):
                    action[k] = action[k].squeeze(1).cpu().numpy()
        else:
            if isinstance(action, torch.Tensor):
                # Squeeze the dimension for discrete action
                action = action.squeeze(1).cpu().numpy()
            action = action[None, :]

        obs, reward, done, info = self.env.step(action.squeeze(0))

        if isinstance(obs, dict):
            for k in obs:
                obs[k] = torch.from_numpy(obs[k]).float().to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)

        reward = torch.from_numpy(reward).unsqueeze(dim=1).float().to(self.device)
        done = torch.from_numpy(done).unsqueeze(dim=1).float().to(self.device)

        return obs, reward, done, info


class Monitor(gym.Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, info_keywords=()):
        super(Monitor, self).__init__(env)
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": time.time(), 'env_id': env.spec and env.spec.id},
                extra_keys=info_keywords)
        else:
            self.results_writer = None
        self.info_keywords = info_keywords
        self.rewards = None

    def reset(self, **kwargs):
        self.reset_state()
        return self.env.reset(**kwargs)

    def reset_state(self):
        self.rewards = []

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return ob, rew, done, info

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info[k]
            if self.results_writer:
                self.results_writer.write_row(epinfo)
            assert isinstance(info, dict)
            if isinstance(info, dict):
                info['episode'] = epinfo

    def close(self):
        super(Monitor, self).close()
        if self.f is not None:
            self.f.close()


class BatchedMonitor(gym.Wrapper):
    """Class to log BatchEnv data."""

    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, info_keywords=()):
        super(BatchedMonitor, self).__init__(env)
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": time.time(), 'env_id': env.spec and env.spec.id},
                extra_keys=info_keywords)
        else:
            self.results_writer = None
        self.info_keywords = info_keywords
        self.rewards = None
        self.steps = None

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def reset_single_env(self, num_env):
        return self.env.reset_single_env(num_env)

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return ob, rew, done, info

    def update(self, ob, rew, done, info):

        if self.rewards is None:
            self.rewards = np.zeros_like(rew, dtype=np.float64)
            self.steps = np.zeros_like(rew, dtype=np.float64)
        else:
            self.rewards += rew
            self.steps += + 1.0

        for num in np.nonzero(done)[0]:
            eprew = float(self.rewards[num])
            eplen = float(self.steps[num])
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                if isinstance(info[k][num], (str, bool, np.bool_)):
                    epinfo[k] = info[k][num]
                else:
                    epinfo[k] = float(info[k][num])

            if self.results_writer:
                self.results_writer.write_row(epinfo)

        info["r"] = copy(self.rewards)
        info["l"] = copy(self.steps)

        self.steps *= (1 - done)
        self.rewards *= (1 - done)

    def close(self):
        super(BatchedMonitor, self).close()
        if self.f is not None:
            self.f.close()


class ResultsWriter(object):
    def __init__(self, filename, header='', extra_keys=()):
        self.extra_keys = extra_keys
        assert filename is not None
        if not filename.endswith(Monitor.EXT):
            if osp.isdir(filename):
                filename = osp.join(filename, Monitor.EXT)
            else:
                filename = filename + "." + Monitor.EXT

        already_exists = osp.isfile(filename)
        self.f = open(filename, "a+")
        if not already_exists:
            if isinstance(header, dict):
                header = '# {} \n'.format(json.dumps(header))
            self.f.write(header)
        self.f.flush()
        self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+tuple(extra_keys))
        if not already_exists:
            self.logger.writeheader()
        self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()


def load_results(dir):
    import pandas
    monitor_files = (
        glob(osp.join(dir, "*monitor.json")) +
        glob(osp.join(dir, "*monitor.csv")))  # get both csv and (old) json files
    if not monitor_files:
        raise Exception("no monitor files of the form *%s found in %s" % (Monitor.EXT, dir))
    dfs = []
    headers = []
    for fname in monitor_files:
        with open(fname, 'rt') as fh:
            if fname.endswith('csv'):
                firstline = fh.readline()
                if not firstline:
                    continue
                assert firstline[0] == '#'
                header = json.loads(firstline[1:])
                df = pandas.read_csv(fh, index_col=None)
                headers.append(header)
            elif fname.endswith('json'):  # Deprecated json format
                episodes = []
                lines = fh.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                df = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            df['t'] += header['t_start']
        dfs.append(df)
    df = pandas.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    df.headers = headers  # HACK to preserve backwards compatibility
    return df
