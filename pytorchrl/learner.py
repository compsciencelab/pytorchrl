import os
import ray
import time
from functools import partial
from collections import defaultdict, deque
from torch.utils.tensorboard import SummaryWriter
import pytorchrl as prl


class Learner:
    """
    Task learner class.

    Class to manage the training process. It pushes forward the training
    process by calling the update workers and tracks progress.

    Parameters
    ----------
    scheme : Scheme
        Training scheme class instance, handling coordination of workers.
    target_steps : int
        Number of environment steps to reach to complete training.
    log_dir : str
        Target directory for model checkpoints and, if specified, Tensorboard logs.
    log_to_tensorboard : bool
        Whether or not generating Tensorboard logs.
    """

    def __init__(self, scheme, target_steps, log_dir=None, log_to_tensorboard=False):

        # Input attributes
        self.log_dir = log_dir
        self.target_steps = target_steps
        self.update_worker = scheme.update_worker()

        # Counters and metrics
        self.num_samples_collected = 0
        self.metrics = {k: defaultdict(partial(deque, maxlen=1)) for k in prl.INFO_KEYS}

        # Define summary writer
        if log_dir and log_to_tensorboard:
            tb_log_dir = "{}/tensorboard_logs".format(log_dir)
            os.makedirs(tb_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_log_dir)
        else: self.writer = None

        # Record starting time
        self.start = time.time()

    def step(self):
        """Takes a logical synchronous optimization step."""

        # Update step
        info = self.update_worker.step()

        actor_version = info[prl.VERSION]

        # grad_update_lag
        info[prl.SCHEME] = {}
        info[prl.SCHEME][prl.FPS] = int(self.num_samples_collected / (time.time() - self.start))
        info[prl.SCHEME][prl.PL] = actor_version[prl.GRADIENT] - actor_version[prl.COLLECTION]
        info[prl.SCHEME][prl.GA] = actor_version[prl.UPDATE] - actor_version[prl.GRADIENT]

        # Update counters
        self.num_samples_collected += info.pop(prl.NUMSAMPLES)

        # Update and log metrics
        for k in prl.INFO_KEYS:
            if k in info and isinstance(info[k], dict):
                for x, y in info[k].items():
                    if isinstance(y, (float, int)):
                        self.metrics[k][x].append(y)
                    if self.writer and isinstance(y, (float, int)):
                        self.writer.add_scalar(os.path.join(
                            k, x), y, self.num_samples_collected)

    def done(self):
        """
        Return True if training has finished (target_steps reached).

        Returns
        -------
        flag : bool
            True if training has reached the target number of steps.
        """
        flag = self.num_samples_collected >= self.target_steps
        if flag:
            self.update_worker.stop()
            print("\nTraining finished!")
            time.sleep(1)
        return flag

    def get_metrics(self, add_algo_metrics=True, add_episodes_metrics=False, add_scheme_metrics=False, add_time_metrics=False):
        """Returns current value of tracked metrics."""

        m = {}

        def include_info(key):
            m[key] = {}
            for k, v in self.metrics[key].items():
                m[os.path.join(key, k)] = sum(v) / len(v)

        if add_algo_metrics:
            include_info(prl.ALGORITHM)

        if add_episodes_metrics:
            include_info(prl.EPISODES)

        if add_scheme_metrics:
            include_info(prl.SCHEME)

        if add_time_metrics:
            include_info(prl.TIME)

        return m

    def print_info(self, add_algo_info=True, add_episodes_info=True, add_scheme_info=False, add_time_info=False):
        """Print relevant information about the training process"""

        def write_info(msg, key):
            msg += "\n  {}: ".format(key)
            for k, v in self.metrics[key].items():
                msg += "{} {:.4f}, ".format(k, sum(v) / len(v))
            msg = msg[:-2]
            return msg

        s = "Update {}".format(self.update_worker.actor_version)
        s += ", num samples collected {}, FPS {}".format(self.num_samples_collected,
            int(self.num_samples_collected / (time.time() - self.start)))

        if add_algo_info:
            s = write_info(s, prl.ALGORITHM)

        if add_episodes_info:
            s = write_info(s, prl.EPISODES)

        if add_scheme_info:
            s = write_info(s, prl.SCHEME)

        if add_time_info:
            s = write_info(s, prl.TIME)

        print(s, flush=True)

    def update_algorithm_parameter(self, parameter_name, new_parameter_value):
        """
        If `parameter_name` is an attribute of the algorithm used for training,
        change its value to `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Worker.algo attribute name
        new_parameter_value : int or float
            New value for `parameter_name`.
        """
        self.update_worker.update_algorithm_parameter(parameter_name, new_parameter_value)

    def save_model(self):
        """
        Save currently learned actor_critic version.

        Returns
        -------
        save_name : str
            Path to saved file.
        """
        fname = os.path.join(self.log_dir, "model.state_dict")
        save_name = self.update_worker.save_model(fname)
        return save_name
