import torch
import csv
import time
import json
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

    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=(), info_keywords=()):
        super(Monitor, self).__init__(env)
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": time.time(), 'env_id': env.spec and env.spec.id},
                extra_keys=reset_keywords + info_keywords)
        else:
            self.results_writer = None
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0

    def reset(self, **kwargs):
        self.reset_state()
        return self.env.reset(**kwargs)

    def reset_state(self):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        for o, r, d, i in zip(ob, rew, done, info):
            self.update(o, r, d, i)
        return ob, rew, done, info

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            if self.results_writer:
                self.results_writer.write_row(epinfo)
            assert isinstance(info, dict)
            if isinstance(info, dict):
                info['episode'] = epinfo

        self.total_steps += 1

    def close(self):
        super(Monitor, self).close()
        if self.f is not None:
            self.f.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times


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