"""
Extension of PPOD that uses intrinsic rewards instead of value predictions to rank demos.
This version does not differentiate between human demos and agent demos, and classifies everything as reward demos.
"""

import os
import glob
import copy
import uuid
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict

import pytorchrl as prl
from pytorchrl.utils import RunningMeanStd
from pytorchrl.agent.storages.on_policy.ppod2_buffer import PPOD2Buffer as B

MIN_BUFFER_SIZE = 10


class PPOD2RebelBuffer(B):
    """
    Storage class for a modified version of the PPO+D algorithm.

    This version of the PPO+D buffer uses intrinsic rewards (if available) instead of value predictions to rank
    potentially interesting demos. Additionally, this version does not differentiate between human demos and agent
    demos, only between reward demos (independently of their origin) and intrinsic demos. If no intrinsic rewards
    are available, only reward demos are replayed.

    Parameters
    ----------
    size : int
        Storage capacity along time axis.
    device: torch.device
        CPU or specific GPU where data tensors will be placed and class
        computations will take place. Should be the same device where the
        actor model is located.
    actor : Actor
        Actor class instance.
    algorithm : Algorithm
        Algorithm class instance.
    envs : VecEnv
        Vector of experiments instance.
    general_value_net_factory : func
        Method to create the reference value function.
    rho : float
        PPO+D rho parameter.
    phi : float
        PPO+D phi parameter.
    gae_lambda : float
        GAE lambda parameter.
    alpha : float
        PPO+D alpha parameter
    initial_reward_threshold : float
        initial intrinsic to use as reward threshold for new reward_demos.
    initial_reward_demos_dir : str
        Path to directory containing agent initial demonstrations.
    supplementary_demos_dir : str
        Path to a directory where additional demos can be added after training has started.
        these demos will be incorporated into the buffer as bonus agent demos.
    target_reward_demos_dir : str
        Path to directory where best reward demonstrations should be saved.
    num_reward_demos_to_save : int
        Number of top reward reward_demos to save.
    initial_int_demos_dir : str
        Path to directory containing intrinsic initial demonstrations.
    target_int_demos_dir : str
        Path to directory where best intrinsic demonstrations should be saved.
    num_int_demos_to_save : int
        Number of top intrinsic reward_demos to save.
    total_buffer_demo_capacity : int
        Maximum number of reward_demos to keep between reward and intrinsic reward_demos.
    save_demos_every : int
        Save top reward_demos every  `save_demo_frequency`th data collection.
    demo_dtypes : dict
        data types to use for the reward_demos.
    """

    # Accepted data fields. Inserting other fields will raise AssertionError
    on_policy_data_fields = prl.OnPolicyDataKeys

    # Data tensors to collect for each reward_demos
    demos_data_fields = prl.DemosDataKeys

    def __init__(self, size, device, actor, algorithm, envs, general_value_net_factory, rho=0.05, phi=0.05, gae_lambda=0.95, alpha=10,
                 total_buffer_demo_capacity=50, initial_reward_threshold=None, initial_reward_demos_dir=None,
                 supplementary_demos_dir=None, target_reward_demos_dir=None, num_reward_demos_to_save=None,
                 initial_int_demos_dir=None, target_int_demos_dir=None, num_int_demos_to_save=None, save_demos_every=10,
                 demo_dtypes={prl.OBS: np.float32, prl.ACT: np.float32,  prl.REW: np.float32}):

        super(PPOD2RebelBuffer, self).__init__(
            size, device, actor, algorithm, envs, rho, phi, gae_lambda, alpha, total_buffer_demo_capacity,
            initial_reward_threshold, initial_reward_demos_dir, supplementary_demos_dir, target_reward_demos_dir,
            num_reward_demos_to_save, initial_int_demos_dir, target_int_demos_dir, num_int_demos_to_save,
            save_demos_every, demo_dtypes)

        # Create general value model and move it to device
        self.general_value_net = general_value_net_factory(self.device)

        # Freeze general value model with respect to optimizers
        for p in self.general_value_net.parameters():
            p.requires_grad = False

    @classmethod
    def create_factory(cls, size, general_value_net_factory, rho=0.05, phi=0.05, gae_lambda=0.95, alpha=10,
                       total_buffer_demo_capacity=50, initial_reward_threshold=None, initial_reward_demos_dir=None,
                       supplementary_demos_dir=None, target_reward_demos_dir=None, num_reward_demos_to_save=None,
                       initial_int_demos_dir=None, target_int_demos_dir=None, num_int_demos_to_save=None,
                       save_demos_every=10, demo_dtypes={prl.OBS: np.float32, prl.ACT: np.float32, prl.REW: np.float32}):
        """
        Returns a function that creates PPOD2Buffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.
        rho : float
            PPO+D rho parameter.
        phi : float
            PPO+D phi parameter.
        gae_lambda : float
            GAE lambda parameter.
        alpha : float
            PPO+D alpha parameter
        total_buffer_demo_capacity : int
            Maximum number of reward_demos to keep between reward and intrinsic reward_demos.
            initial_reward_threshold : float
        initial_reward_threshold : float
            initial intrinsic to use as reward threshold for new reward_demos.
        initial_reward_demos_dir : str
            Path to directory containing agent initial demonstrations.
        supplementary_demos_dir : str
            Path to a directory where additional demos can be added after training has started.
            these demos will be incorporated into the buffer as bonus agent demos.
        target_reward_demos_dir : str
            Path to directory where best reward demonstrations should be saved.
        num_reward_demos_to_save : int
            Number of top reward reward_demos to save.
        initial_int_demos_dir : str
            Path to directory containing intrinsic initial demonstrations.
        target_int_demos_dir : str
            Path to directory where best intrinsic demonstrations should be saved.
        num_int_demos_to_save : int
            Number of top intrinsic reward_demos to save.
        total_buffer_demo_capacity : int
            Maximum number of reward_demos to keep between reward and intrinsic reward_demos.
        save_demos_every : int
            Save top reward_demos every  `save_demo_frequency`th data collection.
        demo_dtypes : dict
            data types to use for the reward_demos.

        Returns
        -------
        create_buffer_instance : func
            creates a new PPOD2Buffer class instance.
        """
        def create_buffer_instance(device, actor, algorithm, envs):
            """Create and return a PPOD2Buffer instance."""
            return cls(size, device, actor, algorithm, envs, general_value_net_factory, rho, phi, gae_lambda,
                       alpha, total_buffer_demo_capacity, initial_reward_threshold, initial_reward_demos_dir,
                       supplementary_demos_dir, target_reward_demos_dir, num_reward_demos_to_save,
                       initial_int_demos_dir, target_int_demos_dir, num_int_demos_to_save,
                       save_demos_every, demo_dtypes)
        return create_buffer_instance

    def before_gradients(self):
        """Before updating actor policy model, compute returns and advantages."""

        self.apply_rebel_logic()
        super(PPOD2RebelBuffer, self).before_gradients()
        print("MIN RETURN:", self.data[prl.RET].min())
        print("MAX RETURN:", self.data[prl.RET].max())
        print("AVG RETURN:", self.data[prl.RET].mean())

    def after_gradients(self, batch, info):
        """
        After updating actor policy model, make sure self.step is at 0.

        Parameters
        ----------
        batch : dict
            Data batch used to compute the gradients.
        info : dict
            Additional relevant info from gradient computation.

        Returns
        -------
        info : dict
            info dict updated with relevant info from Storage.
        """

        super(PPOD2RebelBuffer, self).after_gradients(batch, info)

        info.update({
            "max_reward": self.data[prl.REW].max(),
            "min_reward": self.data[prl.REW].min(),
            "avg_reward": self.data[prl.REW].mean(),
        })

        return info

    def apply_rebel_logic(self):
        ref_value = self.predict_reference_value()
        self.modify_rewards(ref_value)

    def compute_cumulative_rewards(self):
        return

    def predict_reference_value(self):
        return self.general_value_net.get_value(
            self.data[prl.OBS].view(-1, *self.data[prl.OBS].shape[2:]),
            self.data[prl.RHS]["value_net1"].view(-1, *self.data[prl.RHS]["value_net1"].shape[2:]),
            self.data[prl.DONE].view(-1, *self.data[prl.DONE].shape[2:]),
        )['value_net1'].view(self.data[prl.DONE].shape)

    def modify_rewards(self, ref_value, thresh=0.1):
        errors = torch.abs(ref_value - self.data[prl.REW])
        self.data[prl.REW][errors < thresh] *= -1

