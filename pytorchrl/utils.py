import os
import yaml
import torch
import shutil
import argparse


def cleanup_log_dir(log_dir):
    """
    Create log directory and remove old files.

    Parameters
    ----------
    log_dir : str
        Path to log directory.
    """
    try:
        shutil.rmtree(os.path.join(log_dir))
    except Exception:
        print("Unable to cleanup log_dir...")
    os.makedirs(log_dir, exist_ok=True)


class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith('yaml') or values.name.endswith('yml'):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f'Unknown argument in config file: {key}')
            namespace.__dict__.update(config)
        else:
            raise ValueError('Configuration file must end with yaml or yml')


def save_argparse(args, filename, exclude=None):
    if filename.endswith('yaml') or filename.endswith('yml'):
        if isinstance(exclude, str):
            exclude = [exclude, ]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        yaml.dump(args, open(filename, 'w'))
    else:
        raise ValueError("Configuration file should end with yaml or yml")


class RunningMeanStd:
    """Class to keep track on the running mean and variance of tensors batches."""

    def __init__(self, epsilon=1e-4, shape=(), device=torch.device("cpu")):
        self.mean = torch.zeros(shape, dtype=torch.float64).to(device)
        self.var = torch.ones(shape, dtype=torch.float64).to(device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        self.mean, self.var, self.count = new_mean, new_var, new_count
