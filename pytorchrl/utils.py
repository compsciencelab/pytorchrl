import os
import shutil

color2num = dict(gray=30, red=31, green=32, yellow=33, blue=34,
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


