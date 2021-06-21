import torch
from pytorchrl.agent.env.openai_baselines_dependencies.vec_env.vec_env import VecEnvWrapper


class RecordEpisodesWrapper(VecEnvWrapper):
    """
    This wrapper...

    best episodes contain observations, action and rewards.
    """
    def __init__(self, venv, device, num_top_eps=1, max_length_eps=None):
        """Return only every `skip`-th frame"""
        super(RecordEpisodesWrapper, self).__init__(venv)
        self.device = device
        self.num_envs = venv.num_envs
        self.best_episodes = {}

    def reset(self):
        """New vec env reset function"""
        obs = self.venv.reset()
        return obs

    def step_async(self, actions):
        """New vec env step_async function"""
        if isinstance(actions, dict):
            for k in actions:
                if isinstance(actions[k], torch.Tensor):
                    actions[k] = actions[k].squeeze(1).cpu().numpy()
        else:
            if isinstance(actions, torch.Tensor):
                # Squeeze the dimension for discrete actions
                actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        """New vec env step_wait function"""
        obs, reward, done, info = self.venv.step_wait()

        if isinstance(obs, dict):
            for k in obs:
                obs[k] = torch.from_numpy(obs[k]).float().to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)

        reward = torch.from_numpy(reward).unsqueeze(dim=1).float().to(self.device)
        done = torch.from_numpy(done).unsqueeze(dim=1).float().to(self.device)

        return obs, reward, done, info


class VecPyTorch(VecEnvWrapper):
    """
    This wrapper turns obs, reward's and done's from numpy arrays to pytorch
    tensors and places them in the specified device, facilitating interaction
    between the environment and the actor critic function approximators (NNs).

    Parameters
    ----------
    venv : VecEnv
        Original vector environment, previous to applying the wrapper.
    device : torch.device
        CPU or specific GPU where obs, reward's and done's are placed after
        being transformed into pytorch tensors.

    Attributes
    ----------
    device : torch.device
        CPU or specific GPU where obs, reward's and done's are placed after
        being transformed into pytorch tensors.
    num_envs : int
        Size of vector environment.

    """
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        self.num_envs = venv.num_envs

    def reset(self):
        """New vec env reset function"""
        obs = self.venv.reset()
        if isinstance(obs, dict):
            for k in obs:
                obs[k] = torch.from_numpy(obs[k]).float().to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        """New vec env step_async function"""
        if isinstance(actions, dict):
            for k in actions:
                if isinstance(actions[k], torch.Tensor):
                    actions[k] = actions[k].squeeze(1).cpu().numpy()
        else:
            if isinstance(actions, torch.Tensor):
                # Squeeze the dimension for discrete actions
                actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        """New vec env step_wait function"""
        obs, reward, done, info = self.venv.step_wait()

        if isinstance(obs, dict):
            for k in obs:
                obs[k] = torch.from_numpy(obs[k]).float().to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)

        reward = torch.from_numpy(reward).unsqueeze(dim=1).float().to(self.device)
        done = torch.from_numpy(done).unsqueeze(dim=1).float().to(self.device)

        return obs, reward, done, info
