import os
import yaml
import shutil
import argparse


color2num = dict(
    gray=30, red=31, green=32, yellow=33, blue=34,
    magenta=35, cyan=36, white=37, crimson=38)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

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
    #parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__ (self, parser, namespace, values, option_string = None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                namespace.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
        else:
            raise ValueError("configuration file must end with yaml or yml")
    


def save_argparse(args,filename,exclude=None):
    if filename.endswith('yaml') or filename.endswith('yml'):
        if isinstance(exclude, str):
            exclude = [exclude,]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        yaml.dump(args, open(filename, 'w'))
    else:
        raise ValueError("Configuration file should end with yaml or yml")
