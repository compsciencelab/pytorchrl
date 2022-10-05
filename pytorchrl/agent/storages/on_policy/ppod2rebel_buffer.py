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
    error_threshold : float
        Minimum value prediction error. Below error_threshold, reward sign is flipped.
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

    def __init__(self, size, device, actor, algorithm, envs, general_value_net_factory, rho=0.05, phi=0.05,
                 error_threshold=0.01, gae_lambda=0.95, alpha=10, total_buffer_demo_capacity=50,
                 initial_reward_threshold=None, initial_reward_demos_dir=None, supplementary_demos_dir=None,
                 target_reward_demos_dir=None, num_reward_demos_to_save=None, initial_int_demos_dir=None,
                 target_int_demos_dir=None, num_int_demos_to_save=None, save_demos_every=10,
                 demo_dtypes={prl.OBS: np.float32, prl.ACT: np.float32,  prl.REW: np.float32}):

        super(PPOD2RebelBuffer, self).__init__(
            size, device, actor, algorithm, envs, rho, phi, gae_lambda, alpha, total_buffer_demo_capacity,
            initial_reward_threshold, initial_reward_demos_dir, supplementary_demos_dir, target_reward_demos_dir,
            num_reward_demos_to_save, initial_int_demos_dir, target_int_demos_dir, num_int_demos_to_save,
            save_demos_every, demo_dtypes)

        # Define minimum value prediction error
        self.error_threshold = error_threshold

        # Create general value model and move it to device
        self.general_value_net = general_value_net_factory(self.device)

        # Freeze general value model with respect to optimizers
        for p in self.general_value_net.parameters():
            p.requires_grad = False

        # Define reward threshold
        self.reward_threshold = 1.0  # initial_reward_threshold

    @classmethod
    def create_factory(cls, size, general_value_net_factory, rho=0.05, phi=0.05, error_threshold=0.01, gae_lambda=0.95,
                       alpha=10, total_buffer_demo_capacity=50, initial_reward_threshold=None,
                       initial_reward_demos_dir=None, supplementary_demos_dir=None, target_reward_demos_dir=None,
                       num_reward_demos_to_save=None, initial_int_demos_dir=None, target_int_demos_dir=None,
                       num_int_demos_to_save=None, save_demos_every=10,
                       demo_dtypes={prl.OBS: np.float32, prl.ACT: np.float32, prl.REW: np.float32}):
        """
        Returns a function that creates PPOD2RebelBuffer instances.

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
            creates a new PPOD2RebelBuffer class instance.
        """
        def create_buffer_instance(device, actor, algorithm, envs):
            """Create and return a PPOD2RebelBuffer instance."""
            return cls(size, device, actor, algorithm, envs, general_value_net_factory, rho, phi, error_threshold,
                       gae_lambda, alpha, total_buffer_demo_capacity, initial_reward_threshold,
                       initial_reward_demos_dir, supplementary_demos_dir, target_reward_demos_dir,
                       num_reward_demos_to_save, initial_int_demos_dir, target_int_demos_dir, num_int_demos_to_save,
                       save_demos_every, demo_dtypes)
        return create_buffer_instance

    def init_tensors(self, sample):
        """
        Lazy initialization of data tensors from a sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)"""

        super(PPOD2RebelBuffer, self).init_tensors(sample)

        # Add cumulative rewards tensor to self.data
        self.data["CumRew"] = torch.zeros_like(self.data[prl.REW])

    def before_gradients(self):
        """Before updating actor policy model, compute returns and advantages."""

        self.compute_cumulative_rewards()
        self.apply_rebel_logic()
        super(PPOD2RebelBuffer, self).before_gradients()

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

        info = super(PPOD2RebelBuffer, self).after_gradients(batch, info)

        info["Algorithm"].update({
            "max_modified_reward": self.data[prl.REW].max().item(),
            "min_modified_reward": self.data[prl.REW].min().item(),
            "avg_modified_reward": self.data[prl.REW].mean().item(),
        })

        return info

    def apply_rebel_logic(self):
        ref_value = self.predict_reference_value()
        self.modify_rewards(ref_value)

    def compute_cumulative_rewards(self):
        """Compute cumulative episode rewards and also returns."""
        length = self.step - 1 if self.step != 0 else self.max_size
        self.data[prl.RET][length].copy_(self.data[prl.VAL][length])
        self.data["CumRew"].copy_(self.data[prl.REW])
        for step in reversed(range(length)):
            self.data[prl.RET][step] = (self.data[prl.RET][step + 1] * self.algo.gamma * (
                    1.0 - self.data[prl.DONE][step + 1]) + self.data[prl.REW][step])
        for step in range(length + 1):
            self.data["CumRew"][step] += self.data["CumRew"][step - 1] * (1.0 - self.data[prl.DONE][step])

    def predict_reference_value(self):
        """Generate value predictions with the reference value network."""
        return self.general_value_net.get_value(
            self.data[prl.OBS].view(-1, *self.data[prl.OBS].shape[2:]),
            self.data[prl.RHS]["value_net1"].view(-1, *self.data[prl.RHS]["value_net1"].shape[2:]),
            self.data[prl.DONE].view(-1, *self.data[prl.DONE].shape[2:]),
        )['value_net1'].view(self.data[prl.DONE].shape)

    def modify_rewards(self, ref_value):
        """Flip the sign of the rewards with lower reference value prediction than self.error_threshold."""
        errors = torch.abs(ref_value - self.data[prl.RET])
        mask = (errors < self.error_threshold) * (self.data["CumRew"] >= self.reward_threshold) * (
                self.data[prl.REW] > 0.0)
        self.data[prl.REW][mask] *= -1

    def validate_demo(self, demo):
        """
        Verify demo has high value prediction error for rewarded states with
        cumulative reward higher than self.reward_threshold - 1.
        """

        # Compute demo cumulative rewards
        cumulative_rewards = np.copy(demo[prl.REW])
        for step in range(1, len(cumulative_rewards)):
            cumulative_rewards[step] += cumulative_rewards[step - 1]

        # Define mask to get only final states
        mask = (cumulative_rewards >= self.reward_threshold).reshape(-1)

        # Compute demo returns
        returns = np.copy(demo[prl.REW][mask])
        for step in reversed(range(len(returns) - 1)):
            cumulative_rewards[step] += cumulative_rewards[step + 1] * self.algo.gamma

        # Compute reference value prediction for rewarded states with cumulative rewards > self.reward_threshold - 1
        value_preds = self.general_value_net.get_value(
            torch.tensor(demo[prl.OBS][mask]).to(self.device),
            torch.zeros_like(torch.tensor(demo[prl.REW][mask]).to(self.device)),
            torch.zeros_like(torch.tensor(demo[prl.REW][mask])).to(self.device)
        )['value_net1'].cpu().numpy()

        # Verify all predicted errors are higher than self.error_threshold
        valid = (np.abs(value_preds - returns) < self.error_threshold).all()

        return valid

    def track_potential_demos(self, sample):
        """ Tracks current episodes looking for potential agent_demos """

        for i in range(self.num_envs):

            # Copy transition
            for tensor in self.demos_data_fields:
                if tensor in (prl.OBS, ):
                    self.potential_demos["env{}".format(i + 1)][tensor].append(copy.deepcopy(
                        sample[tensor][i, -self.num_channels_obs:]).cpu().numpy().astype(self.demo_dtypes[tensor]))
                else:
                    self.potential_demos["env{}".format(i + 1)][tensor].append(
                        copy.deepcopy(sample[tensor][i]).cpu().numpy().astype(self.demo_dtypes[tensor]))

            # Update cumulative intrinsic reward
            if prl.IREW in sample.keys():

                # Track the cumulative sum of intrinsic rewards of the demo
                self.potential_demos_cumsum_int["env{}".format(i + 1)] += sample[prl.IREW][i].cpu().item()

                # Track the max intrinsic reward of the demo
                self.potential_demos_max_int["env{}".format(i + 1)] = max(
                    self.potential_demos_max_int["env{}".format(i + 1)], sample[prl.IREW][i].cpu().item())

            # Handle end of episode
            if sample[prl.DONE2][i] == 1.0:

                # Get candidate agent_demos
                potential_demo = {}
                for tensor in self.demos_data_fields:
                    potential_demo[tensor] = np.stack(self.potential_demos["env{}".format(i + 1)][tensor])
                    if tensor == prl.REW:
                        nonzero = np.flatnonzero(potential_demo[tensor] > 0.0)
                        last_reward = len(potential_demo[tensor]) if len(nonzero) == 0 else np.max(nonzero)

                # Compute accumulated reward
                episode_reward = potential_demo[prl.REW].sum()
                potential_demo["ID"] = str(uuid.uuid4())
                potential_demo["TotalReward"] = episode_reward
                potential_demo["DemoLength"] = potential_demo[prl.ACT].shape[0]

                # Consider candidate agent_demos for agent_demos reward
                if self.max_reward_demos > 0:

                    if episode_reward >= self.reward_threshold:

                        # Cut off data after last reward
                        for tensor in self.demos_data_fields:
                            potential_demo[tensor] = potential_demo[tensor][0:last_reward + 1]
                        potential_demo["DemoLength"] = potential_demo[prl.ACT].shape[0]

                        # Make sure new reward was previously unknown
                        valid = self.validate_demo(potential_demo)
                        if valid and self.target_reward_demos_dir:
                            filename = "found_demo"
                            save_data = {
                                "Observation": np.array(potential_demo[prl.OBS]).astype(self.demo_dtypes[prl.OBS]),
                                "Reward": np.array(potential_demo[prl.REW]).astype(self.demo_dtypes[prl.REW]),
                                "Action": np.array(potential_demo[prl.ACT]).astype(self.demo_dtypes[prl.ACT]),
                                "FrameSkip": self.frame_skip}
                            np.savez(os.path.join(self.target_reward_demos_dir, filename), **save_data)

                        # Add agent_demos to reward buffer
                        self.reward_demos.append(potential_demo)

                        # Check if buffers are full
                        self.check_demo_buffer_capacity()

                        # Update reward_threshold.
                        self.reward_threshold = min([d["TotalReward"] for d in self.reward_demos])

                        # Update max demo reward
                        self.max_demo_reward = max([d["TotalReward"] for d in self.reward_demos])

                if self.max_intrinsic_demos > 0:

                    # Set potential demo cumulative intrinsic reward
                    episode_ireward = self.eta * self.potential_demos_max_int["env{}".format(i + 1)] + (1 - self.eta) * (
                            self.potential_demos_cumsum_int["env{}".format(i + 1)] / potential_demo["DemoLength"])
                    potential_demo["IntrinsicReward"] = episode_ireward

                    if episode_ireward >= self.intrinsic_threshold or len(self.intrinsic_demos) < self.max_intrinsic_demos:

                        # Add agent_demos to intrinsic buffer
                        self.intrinsic_demos.append(potential_demo)

                        # Check if buffer is full
                        self.check_demo_buffer_capacity()

                        # Update intrinsic_threshold
                        self.intrinsic_threshold = min([p["IntrinsicReward"] for p in self.intrinsic_demos])

                # Reset potential agent_demos dict
                for tensor in self.demos_data_fields:
                    self.potential_demos["env{}".format(i + 1)][tensor] = []
                    self.potential_demos_max_int["env{}".format(i + 1)] = 0.0
                    self.potential_demos_cumsum_int["env{}".format(i + 1)] = 0.0
