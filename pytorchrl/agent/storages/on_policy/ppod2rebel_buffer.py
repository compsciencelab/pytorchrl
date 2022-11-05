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
from pytorchrl.agent.actors.utils import partially_load_checkpoint
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
    reward_predictor_factory : func
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
    validate_entire_demos : bool
        If True, valid demos need not only to visit unknown rewarded states after reaching the reward threshold,
        but also need to visit exclusively known rewarded state before reaching the reward threshold. Known and unknown
        is defined by being over or below the error_threshold for the reward_predictor.
    save_demos_every : int
        Save top reward_demos every  `save_demo_frequency`th data collection.
    demo_dtypes : dict
        Data types to use for the reward_demos.
    """

    # Accepted data fields. Inserting other fields will raise AssertionError
    on_policy_data_fields = prl.OnPolicyDataKeys

    # Data tensors to collect for each reward_demos
    demos_data_fields = prl.DemosDataKeys

    def __init__(self, size, device, actor, algorithm, envs, reward_predictor_factory=None,
                 reward_predictor_net_kwargs={}, restart_reward_predictor_net=None, rho=0.05, phi=0.05, gae_lambda=0.95,
                 alpha=10, total_buffer_demo_capacity=50, initial_reward_threshold=None, initial_reward_demos_dir=None,
                 supplementary_demos_dir=None, target_reward_demos_dir=None, num_reward_demos_to_save=None,
                 initial_int_demos_dir=None, target_int_demos_dir=None, num_int_demos_to_save=None,
                 validate_entire_demos=False, save_demos_every=10,
                 demo_dtypes={prl.OBS: np.float32, prl.ACT: np.float32,  prl.REW: np.float32}):

        super(PPOD2RebelBuffer, self).__init__(
            size, device, actor, algorithm, envs, rho, phi, gae_lambda, alpha, total_buffer_demo_capacity,
            initial_reward_threshold, initial_reward_demos_dir, supplementary_demos_dir, target_reward_demos_dir,
            num_reward_demos_to_save, initial_int_demos_dir, target_int_demos_dir, num_int_demos_to_save,
            save_demos_every, demo_dtypes)

        # Create target demo dirs
        os.makedirs(target_reward_demos_dir, exist_ok=True)

        # Create general value model and move it to device
        if reward_predictor_factory:
            model = torch.nn.Module()
            model.reward_predictor = reward_predictor_factory(**reward_predictor_net_kwargs).to(self.device)
            model.error_threshold = torch.nn.parameter.Parameter(
                data=torch.tensor(-1000000, dtype=torch.float32), requires_grad=False)
            if restart_reward_predictor_net:
                partially_load_checkpoint(
                    model, "reward_predictor", restart_reward_predictor_net, map_location=self.device)
                partially_load_checkpoint(
                    model, "error_threshold", restart_reward_predictor_net, map_location=self.device)

            # Freeze general value model with respect to optimizers
            for p in model.reward_predictor.parameters():
                p.requires_grad = False

            self.actor.predictor = model

        else:
            self.actor.predictor = None

        self.validate_entire_demos = validate_entire_demos

        # Define reward and error threshold
        self.reward_threshold = self.initial_reward_threshold = initial_reward_threshold or 0.0

    @classmethod
    def create_factory(cls, size, reward_predictor_factory=None, reward_predictor_net_kwargs={},
                       restart_reward_predictor_net=None, rho=0.05, phi=0.05, gae_lambda=0.95, alpha=10,
                       total_buffer_demo_capacity=50, initial_reward_threshold=None, initial_reward_demos_dir=None,
                       supplementary_demos_dir=None, target_reward_demos_dir=None, num_reward_demos_to_save=None,
                       initial_int_demos_dir=None, target_int_demos_dir=None, num_int_demos_to_save=None,
                       validate_entire_demos=False, save_demos_every=10,
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
        validate_entire_demos : bool
            If True, valid demos need not only to visit unknown rewarded states after reaching the reward threshold,
            but also need to visit exclusively known rewarded state before reaching the reward threshold. Known and unknown
            is defined by being over or below the error_threshold for the reward_predictor.
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
            return cls(size, device, actor, algorithm, envs, reward_predictor_factory, reward_predictor_net_kwargs,
                       restart_reward_predictor_net, rho, phi, gae_lambda, alpha, total_buffer_demo_capacity,
                       initial_reward_threshold, initial_reward_demos_dir, supplementary_demos_dir,
                       target_reward_demos_dir, num_reward_demos_to_save, initial_int_demos_dir, target_int_demos_dir,
                       num_int_demos_to_save, validate_entire_demos, save_demos_every, demo_dtypes)
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
        if self.actor.predictor:
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
        self.data["CumRew"].copy_(self.data[prl.REW])
        for step in range(length + 1):
            self.data["CumRew"][step] += self.data["CumRew"][step - 1] * (1.0 - self.data[prl.DONE][step])

    def predict_reference_value(self):
        """Generate value predictions with the reference value network."""
        return self.actor.predictor.reward_predictor(
            self.data[prl.OBS].view(-1, *self.data[prl.OBS].shape[2:])
        ).view(self.data[prl.DONE].shape)

    def modify_rewards(self, ref_value):
        """Flip the sign of the rewards with lower reference value prediction than self.actor.error_threshold."""
        errors = torch.abs(ref_value - self.data[prl.REW])
        mask = (errors < float(self.actor.predictor.error_threshold)) * (self.data["CumRew"] > float(
            self.reward_threshold)) * (self.data[prl.REW] > 0.0)
        self.data[prl.REW][mask] *= -1

    def validate_demo(self, demo):
        """
        Verify demo has high value prediction error for rewarded states with
        cumulative reward higher than self.reward_threshold - 1.
        """

        if not self.actor.predictor:
            return True

        # Compute demo cumulative rewards
        cumulative_rewards = np.cumsum(demo[prl.REW], axis=0)[self.frame_stack - 1:]

        # Get demo rewards
        rewards = np.copy(demo[prl.REW])[self.frame_stack - 1:]

        # Define mask to get only final states
        mask1 = np.logical_and(cumulative_rewards > self.initial_reward_threshold, rewards != 0)  # after thresh
        mask2 = np.logical_and(cumulative_rewards <= self.initial_reward_threshold, rewards != 0)  # from 0 to thresh

        # Create stacked observations
        stacked_obs = []
        for start in range(self.frame_stack):
            end = - (self.frame_stack - 1 - start)
            if end == 0:
                end = None
            stacked_obs.append(demo[prl.OBS][start:end])
        stacked_obs = np.concatenate(stacked_obs, axis=1)

        # Compute reward prediction for rewarded states with cumulative rewards < self.reward_threshold
        reward_preds = self.actor.predictor.reward_predictor(torch.tensor(stacked_obs).to(self.device)).cpu().numpy()

        # Verify all predicted errors are higher than self.error_threshold at the end and lower at the beginning
        validation1 = (np.abs(reward_preds[mask1] - rewards[mask1]) > float(self.actor.predictor.error_threshold)).all()
        if self.validate_entire_demos:
            validation2 = (np.abs(reward_preds[mask2] - rewards[mask2]) < float(
                self.actor.predictor.error_threshold)).all()
        else:
            validation2 = True
        validation = validation1 and validation2

        return validation

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

                        if valid:
                            
                            # Add agent_demos to reward buffer
                            self.reward_demos.append(potential_demo)

                            # Check if buffers are full
                            self.check_demo_buffer_capacity()

                            # Update reward_threshold.
                            self.reward_threshold = max(
                                self.reward_threshold, min([d["TotalReward"] for d in self.reward_demos]))
                            
                            # Update max demo reward
                            self.max_demo_reward = max([d["TotalReward"] for d in self.reward_demos])

                if self.max_intrinsic_demos > 0:

                    # Set potential demo cumulative intrinsic reward
                    episode_ireward = self.eta * self.potential_demos_max_int["env{}".format(i + 1)] + (1 - self.eta) * (
                            self.potential_demos_cumsum_int["env{}".format(i + 1)] / potential_demo["DemoLength"])
                    potential_demo["IntrinsicReward"] = episode_ireward

                    if episode_ireward >= self.intrinsic_threshold or \
                            len(self.intrinsic_demos) < self.max_intrinsic_demos:

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
