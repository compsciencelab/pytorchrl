import os
import yaml
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
