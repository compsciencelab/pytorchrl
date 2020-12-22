import os
import shutil

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


