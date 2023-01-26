import os
import glob
import copy
import uuid
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict

import pytorchrl as prl
from pytorchrl.agent.storages.on_policy.gae_buffer import GAEBuffer as B

MIN_BUFFER_SIZE = 10


class PPOD2Buffer(B):
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
    storage_tensors = prl.OnPolicyDataKeys

    # Data tensors to collect for each reward_demos
    demos_data_fields = prl.DemosDataKeys

    def __init__(self, size, device, actor, algorithm, envs, rho=0.05, phi=0.05, gae_lambda=0.95, alpha=10,
                 total_buffer_demo_capacity=50, initial_reward_threshold=None, initial_reward_demos_dir=None,
                 supplementary_demos_dir=None, target_reward_demos_dir=None, num_reward_demos_to_save=None,
                 initial_int_demos_dir=None, target_int_demos_dir=None, num_int_demos_to_save=None, save_demos_every=10,
                 demo_dtypes={prl.OBS: np.float32, prl.ACT: np.float32,  prl.REW: np.float32}):

        super(PPOD2Buffer, self).__init__(
            size=size,
            envs=envs,
            actor=actor,
            device=device,
            algorithm=algorithm,
            gae_lambda=gae_lambda,
        )

        # PPO + D parameters
        self.rho = rho
        self.phi = phi
        self.alpha = alpha
        self.iter = 0
        self.eta = 0.9  # To weight max and cumulative intrinsic rewards

        # Define buffer demos capacity
        if rho != 0.0 or phi != 0.0:
            self.max_reward_demos = max(MIN_BUFFER_SIZE, int(total_buffer_demo_capacity * (rho / (rho + phi))))
            self.max_intrinsic_demos = max(MIN_BUFFER_SIZE, int(total_buffer_demo_capacity * (phi / (rho + phi))))
        else:
            self.max_reward_demos = MIN_BUFFER_SIZE
            self.max_intrinsic_demos = MIN_BUFFER_SIZE

        # Demo - related parameters
        self.save_demos_every = save_demos_every
        self.max_demos = total_buffer_demo_capacity
        self.num_reward_demos_to_save = num_reward_demos_to_save or self.max_reward_demos
        self.initial_reward_demos_dir = initial_reward_demos_dir
        self.target_reward_demos_dir = target_reward_demos_dir
        self.num_int_demos_to_save = num_int_demos_to_save or self.max_intrinsic_demos
        self.initial_int_demos_dir = initial_int_demos_dir
        self.supplementary_demos_dir = supplementary_demos_dir
        self.target_int_demos_dir = target_int_demos_dir

        # Data parameters
        self.demo_dtypes = demo_dtypes
        self.num_channels_obs = None  # Lazy initialization
        self.inserted_samples = 0
        self.storage_tensors += (prl.MASK, )

        # Define reward and intrinsic buffers
        self.reward_demos = []
        self.intrinsic_demos = []

        # Load initial reward_demos
        self.load_initial_demos()

        # Define reward_threshold
        self.reward_threshold = initial_reward_threshold or - np.inf
        if len(self.reward_demos) > 0:
            self.reward_threshold = max(
                self.reward_threshold, min([d["TotalReward"] for d in self.reward_demos]))
        self.max_demo_reward = max(
            [d["TotalReward"] for d in self.reward_demos]) if len(self.reward_demos) > 0 else -np.inf

        # Define intrinsic_threshold
        self.intrinsic_threshold = - np.inf
        if len(self.intrinsic_demos) > 0:
            self.intrinsic_threshold = min([d["IntrinsicReward"] for d in self.intrinsic_demos])

        # Define variables to track potentially interesting demos
        self.potential_demos_cumsum_int = {"env{}".format(i + 1): 0.0 for i in range(self.num_envs)}
        self.potential_demos_max_int = {"env{}".format(i + 1): 0.0 for i in range(self.num_envs)}
        self.potential_demos = {"env{}".format(i + 1): defaultdict(list) for i in range(self.num_envs)}

        # Define variable to track reward_demos in progress
        self.demos_in_progress = {
            "env{}".format(i + 1): {
                "ID": None, "Demo": None, "Step": 0, "DemoLength": -1, "DemoType": None,
                "CumIntrinsicReward": 0.0, "MaxIntrinsicReward": 0.0, prl.RHS: None,
            } for i in range(self.num_envs)}

        # To keep track of supplementary demos loaded
        self.supplementary_demos_loaded = []

    @classmethod
    def create_factory(cls, size, rho=0.05, phi=0.05, gae_lambda=0.95, alpha=10, total_buffer_demo_capacity=50,
                       initial_reward_threshold=None, initial_reward_demos_dir=None,
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
            return cls(size, device, actor, algorithm, envs, rho, phi, gae_lambda, alpha,
                       total_buffer_demo_capacity, initial_reward_threshold, initial_reward_demos_dir,
                       supplementary_demos_dir, target_reward_demos_dir, num_reward_demos_to_save,
                       initial_int_demos_dir, target_int_demos_dir, num_int_demos_to_save,
                       save_demos_every, demo_dtypes)
        return create_buffer_instance

    def init_tensors(self, sample):
        """
        Lazy initialization of data tensors from a sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """

        if prl.EMBED in sample.keys():
            self.demos_data_fields += (prl.EMBED,)
            self.storage_tensors += (prl.EMBED,)
            self.demo_dtypes[prl.EMBED] = sample[prl.EMBED].cpu().numpy().dtype

        super(PPOD2Buffer, self).init_tensors(sample)

        if prl.IREW not in sample.keys():
            self.phi = 0.0
            self.max_reward_demos = self.max_demos
            self.max_intrinsic_demos = 0

    def before_gradients(self):
        """Before updating actor policy model, compute returns and advantages."""

        super(PPOD2Buffer, self).before_gradients()
        print("\nREWARD DEMOS {}, INTRINSIC DEMOS {}, RHO {}, PHI {}, REWARD THRESHOLD {}, MAX DEMO REWARD {},"
              " INTRINSIC THRESHOLD {}\n".format(len(self.reward_demos), len(self.intrinsic_demos),
            self.rho, self.phi, self.reward_threshold, self.max_demo_reward, self.intrinsic_threshold))

        self.iter += 1
        if self.iter % self.save_demos_every == 0:
            self.save_demos()

        if self.supplementary_demos_dir:
            self.load_supplementary_demos()

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

        super(PPOD2Buffer, self).after_gradients(batch, info)

        # info['NumberSamples'] -= self.inserted_samples
        self.inserted_samples = 0

        return info

    def insert_transition(self, sample):
        """
        Store new transition sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """

        # If obs embeddings available, add them (for logging purposes)
        if prl.EMBED in sample[prl.INFO][0].keys():
            obs_embeds = [i[prl.EMBED] for i in sample[prl.INFO]]
            obs_embeds = np.stack(obs_embeds).reshape(len(obs_embeds), -1)
            sample.update({prl.EMBED: torch.from_numpy(obs_embeds)})

        # Data tensors lazy initialization, only executed the first time
        if self.size == 0 and self.data[prl.OBS] is None:
            self.init_tensors(sample)
            self.get_num_channels_obs(sample)

        # Insert environment sample data
        for k in sample:

            if k not in self.storage_tensors:
                continue

            if not self.recurrent_actor and k in (prl.RHS, prl.RHS2):
                continue

            # We use the same tensor to store obs and obs2
            # We also use single tensors for rhs and rhs2,
            # and done and done2
            if k in (prl.OBS, prl.RHS, prl.DONE):
                pos = self.step + 1
                sample_k = "Next" + k
            else:
                pos = self.step
                sample_k = k

            # Copy sample tensor to buffer target position
            if isinstance(sample[k], dict):
                for x, v in sample[k].items():
                    self.data[k][x][pos].copy_(sample[sample_k][x])
            else:
                self.data[k][pos].copy_(sample[sample_k])

        # Track episodes for potential agent_demos
        self.track_potential_demos(sample)

        # Set inserted data mask to 1.0
        if prl.MASK in self.data.keys():
            self.data[prl.MASK][self.step].fill_(1.0)

        # Overwrite demo data in environments where a demo being replayed
        all_envs, all_obs, all_done, all_rhs = [], [], [], {k: [] for k in sample[prl.RHS].keys()}

        # Step 1: prepare tensors
        for i in range(self.num_envs):

            # If demo replay is in progress
            if self.demos_in_progress["env{}".format(i + 1)]["Demo"]:

                # Get demo obs, rhs and done tensors to run forward pass
                obs = self.data[prl.OBS][self.step][i].unsqueeze(0)
                if self.demos_in_progress["env{}".format(i + 1)][prl.RHS]:
                    rhs = self.demos_in_progress["env{}".format(i + 1)][prl.RHS]
                    done = torch.zeros(1, 1).to(self.device)
                else:
                    obs, rhs, done = self.actor.actor_initial_states(obs)

                all_envs.append(i)
                all_obs.append(obs)
                all_done.append(done)
                for k in rhs:
                    all_rhs[k].append(rhs[k])

            # Otherwise check if end of episode reached and randomly start new demo
            elif sample[prl.DONE2][i] == 1.0:

                self.sample_demo(env_id=i)

        # Step 2: perform acting step and overwrite data
        if len(all_envs) > 0:

            # Cast obs, rhs, dones into the right format
            all_obs = torch.cat(all_obs)
            all_done = torch.cat(all_done)
            for k in all_rhs:
                all_rhs[k] = torch.cat(all_rhs[k])

            # Run forward pass
            _, _, rhs2, algo_data = self.algo.acting_step(all_obs, all_rhs, all_done)

            for num, i in enumerate(all_envs):

                demo_step = self.demos_in_progress["env{}".format(i + 1)]["Step"]

                # Insert demo act and rew tensors to self.step
                for tensor in (prl.ACT, prl.REW):
                    self.data[tensor][self.step][i].copy_(
                        torch.FloatTensor(self.demos_in_progress["env{}".format(i + 1)]["Demo"][tensor][demo_step]))

                # Insert demo logprob to self.step. Demo action prob is 1.0, so logprob is 0.0
                self.data[prl.LOGP][self.step][i].copy_(torch.zeros(1))

                # Set inserted data mask to 0.0 only if Intrinsic demo
                if prl.MASK in self.data.keys():
                    if self.demos_in_progress["env{}".format(i + 1)]["DemoType"] == "Intrinsic":
                        self.data[prl.MASK][self.step][i].copy_(torch.zeros(1))

                # Insert other tensors predicted in the forward pass
                for tensor in (prl.IREW, prl.VAL, prl.IVAL):
                    if tensor in algo_data.keys():
                        self.data[tensor][self.step][i].copy_(algo_data[tensor][num])

                    if tensor == prl.IREW and prl.IREW in algo_data.keys():
                        # Track the cumulative sum of intrinsic rewards of the demo
                        self.demos_in_progress["env{}".format(i + 1)]["CumIntrinsicReward"] += \
                            algo_data[prl.IREW][num].item()
                        # Track the max intrinsic reward of the demo
                        self.demos_in_progress["env{}".format(i + 1)]["MaxIntrinsicReward"] = max(
                            self.demos_in_progress["env{}".format(i + 1)]["MaxIntrinsicReward"],
                            algo_data[prl.IREW][num].item())

                # Update demo_in_progress variables
                self.demos_in_progress["env{}".format(i + 1)]["Step"] += 1
                self.demos_in_progress["env{}".format(i + 1)][prl.RHS] = {
                    k: rhs2[k][num].reshape(1, -1) for k in rhs2.keys()}
                self.inserted_samples += 1

                # Handle end of demo
                if demo_step == self.demos_in_progress["env{}".format(i + 1)]["DemoLength"] - 1:

                    # If intrinsic demo
                    if self.demos_in_progress["env{}".format(i + 1)]["DemoType"] == "Intrinsic":
                        for intrinsic_demo in self.intrinsic_demos:
                            # If demo still in buffer, update IntrinsicReward
                            if self.demos_in_progress["env{}".format(i + 1)]["Demo"]["ID"] == intrinsic_demo["ID"]:
                                intrinsic_demo["IntrinsicReward"] = \
                                    self.eta * self.demos_in_progress["env{}".format(i + 1)]["MaxIntrinsicReward"] + \
                                    (1 - self.eta) * (self.demos_in_progress["env{}".format(i + 1)]["CumIntrinsicReward"] /
                                                      self.demos_in_progress["env{}".format(i + 1)]["DemoLength"])

                    # Randomly sample new demo if last demo has finished
                    self.sample_demo(env_id=i)

                else:

                    # Insert demo done2 tensor to self.step + 1
                    self.data[prl.DONE][self.step + 1][i].copy_(torch.zeros(1))

                    # Insert demo obs2 tensor to self.step + 1
                    obs2 = torch.roll(all_obs[num:num + 1], -self.num_channels_obs, dims=1).squeeze(0)
                    obs2[-self.num_channels_obs:].copy_(torch.FloatTensor(
                        self.demos_in_progress["env{}".format(i + 1)]["Demo"][prl.OBS][demo_step + 1]))
                    self.data[prl.OBS][self.step + 1][i].copy_(obs2)

                    # Insert demo rhs2 tensor to self.step + 1
                    if self.recurrent_actor:
                        for k in self.data[prl.RHS]:
                            self.data[prl.RHS][k][self.step + 1][i].copy_(rhs2[k][num].squeeze())

        self.step = (self.step + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

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

                # Get candidate demo
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

                # Consider candidate demo to be a reward demo
                if self.max_reward_demos > 0:

                    if episode_reward >= self.reward_threshold:

                        # Cut off data after last reward
                        for tensor in self.demos_data_fields:
                            potential_demo[tensor] = potential_demo[tensor][0:last_reward + 1]
                        potential_demo["DemoLength"] = potential_demo[prl.ACT].shape[0]

                        # Add demo to reward buffer
                        self.reward_demos.append(potential_demo)

                        # Check if buffers are full
                        self.check_demo_buffer_capacity()

                        # Update reward_threshold.
                        self.reward_threshold = max(
                            self.reward_threshold, min([d["TotalReward"] for d in self.reward_demos]))

                        # Update max demo reward
                        self.max_demo_reward = max([d["TotalReward"] for d in self.reward_demos])

                # Consider candidate demo to be a intrinsic demo
                if self.max_intrinsic_demos > 0:

                    # Set potential demo cumulative intrinsic reward
                    episode_ireward = self.eta * self.potential_demos_max_int["env{}".format(i + 1)] + (1 - self.eta) * (
                            self.potential_demos_cumsum_int["env{}".format(i + 1)] / potential_demo["DemoLength"])
                    potential_demo["IntrinsicReward"] = episode_ireward

                    if episode_ireward >= self.intrinsic_threshold or len(self.intrinsic_demos) < self.max_intrinsic_demos:

                        # Add demo to intrinsic buffer
                        self.intrinsic_demos.append(potential_demo)

                        # Check if buffer is full
                        self.check_demo_buffer_capacity()

                        # Update intrinsic_threshold
                        self.intrinsic_threshold = min([p["IntrinsicReward"] for p in self.intrinsic_demos])

                # Reset potential demo dict
                for tensor in self.demos_data_fields:
                    self.potential_demos["env{}".format(i + 1)][tensor] = []
                    self.potential_demos_max_int["env{}".format(i + 1)] = 0.0
                    self.potential_demos_cumsum_int["env{}".format(i + 1)] = 0.0

    def load_demo(self, demo_path):
        """Loads and returns a environment demonstration."""

        # Load demos tensors
        demo = np.load(demo_path)
        new_demo = {k: {} for k in self.demos_data_fields}

        if int(demo["FrameSkip"]) != self.frame_skip:
            raise ValueError(
                "Env and demo with different frame skip!")

        # Add action, obs, rew
        new_demo[prl.ACT] = demo[prl.ACT].astype(self.demo_dtypes[prl.OBS])
        new_demo[prl.OBS] = demo[prl.OBS].astype(self.demo_dtypes[prl.REW])
        new_demo[prl.REW] = demo[prl.REW].astype(self.demo_dtypes[prl.ACT])
        if prl.EMBED in demo.keys() and prl.EMBED in self.demos_data_fields:
            new_demo[prl.EMBED] = demo[prl.EMBED].astype(self.demo_dtypes[prl.EMBED])

        new_demo.update({
            "ID": str(uuid.uuid4()),
            "DemoLength": demo[prl.ACT].shape[0],
            "TotalReward": new_demo[prl.REW].sum(),
            "IntrinsicReward": 1000,
        })

        return new_demo

    def load_initial_demos(self):
        """
        Load initial demonstrations.

        Warning: make sure the frame_skip and frame_stack hyper-parameters are
        the same as those used to record the demonstrations!
        """

        loaded_int_demos, loaded_reward_demos = [], []

        initial_reward_demos = glob.glob(self.initial_reward_demos_dir + '/*.npz') if self.initial_reward_demos_dir else []
        initial_int_demos = glob.glob(self.initial_int_demos_dir + '/*.npz') if self.initial_int_demos_dir else []

        max_reward_demos = max(0, int(self.max_demos * (self.rho / (self.rho + self.phi)))) if self.rho != 0.0 else int(0)
        max_int_demos = int(self.max_demos * (self.phi / (self.rho + self.phi))) if self.phi != 0.0 else int(0)

        for num, demo_file in enumerate(initial_reward_demos):

            if (num + 1) > max_reward_demos:
                continue

            try:

                new_demo = self.load_demo(demo_file)
                self.reward_demos.append(new_demo)
                loaded_reward_demos.append(new_demo["ID"])

            except Exception:
                print("Failed to load agent demo!")

        for num, demo_file in enumerate(initial_int_demos):

            if (num + 1) > max_int_demos:
                continue

            try:

                new_demo = self.load_demo(demo_file)
                self.intrinsic_demos.append(new_demo)
                loaded_int_demos.append(new_demo["ID"])

            except Exception:
                print("Failed to load intrinsic demo!")

        print("\nLOADED {} AGENT DEMOS AND {} INTRINSIC DEMOS".format(len(loaded_reward_demos), len(loaded_int_demos)))

    def load_supplementary_demos(self):
        """
        Load demonstrations found in the self.supplementary_demos (if any).
        Warning: make sure the environment frame_skip and frame_stack hyperparameters are
        the same as those used in the demonstrations!
        """

        # Create supp dir if necessary
        if not os.path.exists(self.supplementary_demos_dir):
            os.makedirs(self.supplementary_demos_dir, exist_ok=True)

        num_loaded_supplementary_demos = 0
        supplementary_demos = glob.glob(self.supplementary_demos_dir + '/*.npz') if self.supplementary_demos_dir else []

        for demo_file in supplementary_demos:

            if demo_file not in self.supplementary_demos_loaded:

                self.supplementary_demos_loaded.append(demo_file)
                try:

                    new_demo = self.load_demo(demo_file)
                    self.reward_demos.append(new_demo)
                    self.intrinsic_demos.append(new_demo)
                    num_loaded_supplementary_demos += 1

                    # Check if buffer is full
                    self.check_demo_buffer_capacity()

                    # Update reward_threshold.
                    self.reward_threshold = max(
                        self.reward_threshold, min([d["TotalReward"] for d in self.reward_demos]))
                    
                    # Update intrinsic_threshold
                    self.intrinsic_threshold = min([p["IntrinsicReward"] for p in self.intrinsic_demos])

                    # Update max demo reward
                    self.max_demo_reward = max([d["TotalReward"] for d in self.reward_demos])

                except Exception:
                    print("Failed to load supplementary demo!")

        # Check if buffer is full
        self.check_demo_buffer_capacity()

        if num_loaded_supplementary_demos > 0:
            print("LOADED {} SUPPLEMENTARY DEMOS\n".format(num_loaded_supplementary_demos))

    def sample_demo(self, env_id):
        """With probability rho insert reward reward_demos, with probability phi insert intrinsic reward_demos."""

        # Reset reward_demos tracking variables
        self.demos_in_progress["env{}".format(env_id + 1)]["Step"] = 0
        self.demos_in_progress["env{}".format(env_id + 1)][prl.RHS] = None

        # Sample episode type
        episode_source = np.random.choice(["reward_demo", "intrinsic_demo", "env"],
            p=[self.rho, self.phi, 1.0 - self.rho - self.phi])

        if episode_source == "reward_demo" and len(self.reward_demos) > 0:

            # Randomly select a reward demo
            selected = np.random.choice(range(len(self.reward_demos)))
            demo = copy.deepcopy(self.reward_demos[selected])
            demo_type = "Reward"

        elif episode_source == "intrinsic_demo" and len(self.intrinsic_demos) > 0:

            # randomly select a intrinsic demo
            probs = np.array([(p["IntrinsicReward"]) ** self.alpha for p in self.intrinsic_demos])
            probs += 1e-10
            probs = (probs / probs.sum()).astype(np.float32)
            selected = np.random.choice(range(len(self.intrinsic_demos)), p=probs)
            demo = copy.deepcopy(self.intrinsic_demos[selected])
            demo_type = "Intrinsic"

        else:
            demo = None
            demo_type = None

        # Define demo and demo_type
        self.demos_in_progress["env{}".format(env_id + 1)]["Demo"] = demo
        self.demos_in_progress["env{}".format(env_id + 1)]["DemoType"] = demo_type

        # Set done to True
        self.data[prl.DONE][self.step + 1][env_id].copy_(torch.ones(1).to(self.device))

        # Set initial rhs to zeros
        if self.recurrent_actor:
            for k in self.data[prl.RHS]:
                self.data[prl.RHS][k][self.step + 1][env_id].fill_(0.0)

        if demo:

            # Set reward_demos length
            self.demos_in_progress["env{}".format(env_id + 1)]["DemoLength"] = demo["DemoLength"]

            # Set reward_demos MaxIntrinsicReward and CumIntrinsicReward to 0.0
            self.demos_in_progress["env{}".format(env_id + 1)]["CumIntrinsicReward"] = 0.0
            self.demos_in_progress["env{}".format(env_id + 1)]["MaxIntrinsicReward"] = 0.0

            # Set next buffer obs to be the starting demo obs
            for k in range(self.frame_stack):
                self.data[prl.OBS][self.step + 1][env_id][
                k * self.num_channels_obs:(k + 1) * self.num_channels_obs].copy_(torch.FloatTensor(
                    self.demos_in_progress["env{}".format(env_id + 1)]["Demo"][prl.OBS][0]))

        else:
            # Reset `i-th` environment as set next buffer obs to be the starting episode obs
            self.data[prl.OBS][self.step + 1][env_id].copy_(self.envs.reset_single_env(env_id=env_id).squeeze())

            # Reset potential reward_demos dict
            for tensor in self.demos_data_fields:
                self.potential_demos["env{}".format(env_id + 1)][tensor] = []
                self.potential_demos_max_int["env{}".format(env_id + 1)] = 0.0
                self.potential_demos_cumsum_int["env{}".format(env_id + 1)] = 0.0

    def check_demo_buffer_capacity(self):
        """
        Check total amount of reward_demos. If total amount of reward_demos exceeds
        self.max_demos, pop reward_demos.
        """

        # First pop intrinsic reward_demos with lowest CumulativeIntrinsicReward
        if len(self.intrinsic_demos) > self.max_intrinsic_demos:
            for _ in range(len(self.intrinsic_demos) - self.max_intrinsic_demos):
                del self.intrinsic_demos[np.array([p["IntrinsicReward"] for p in self.intrinsic_demos]).argmin()]

        # Second pop reward reward_demos with reward lower than self.max_demo_reward
        if len(self.reward_demos) > self.max_reward_demos:
            for _ in range(len(self.reward_demos) - self.max_reward_demos):
                rewards = np.array([p[prl.REW].sum() for p in self.reward_demos])
                if rewards[np.argmin(rewards)] < self.max_demo_reward:
                    del self.reward_demos[np.argmin(rewards)]

        # Finally eject longest reward_demos of those with reward self.max_demo_reward
        if len(self.reward_demos) > self.max_reward_demos:
            for _ in range(len(self.reward_demos) - self.max_reward_demos):
                lengths = np.array([p[prl.OBS].shape[0] for p in self.reward_demos])
                del self.reward_demos[np.argmax(lengths)]

    def save_demos(self):
        """
        Saves the top `num_rewards_demos` agent_demos from the reward agent_demos buffer and
        the top `num_intrinsic_demos` agent_demos from the intrinsic agent_demos buffer.
        """

        if self.target_reward_demos_dir:

            # Create target dir for reward agent_demos if necessary
            if not os.path.exists(self.target_reward_demos_dir):
                os.makedirs(self.target_reward_demos_dir, exist_ok=True)

            # Rank agent agent_demos according to episode reward
            reward_ranking = np.flip(np.array([d["TotalReward"] for d in self.reward_demos]).argsort())

            # Save agent reward agent_demos
            saved_demos = 0
            for demo_pos in reward_ranking:

                if saved_demos == self.num_reward_demos_to_save:
                    break

                filename = "reward_demo_{}".format(saved_demos + 1)

                save_data = {
                    prl.OBS: np.array(self.reward_demos[demo_pos][prl.OBS]).astype(self.demo_dtypes[prl.OBS]),
                    prl.REW: np.array(self.reward_demos[demo_pos][prl.REW]).astype(self.demo_dtypes[prl.REW]),
                    prl.ACT: np.array(self.reward_demos[demo_pos][prl.ACT]).astype(self.demo_dtypes[prl.ACT]),
                    "FrameSkip": self.frame_skip}

                if prl.EMBED in self.data.keys() and prl.EMBED in self.demos_data_fields \
                        and prl.EMBED in self.reward_demos[demo_pos].keys():
                    save_data.update({prl.EMBED: np.array(self.reward_demos[demo_pos][prl.EMBED])})

                np.savez(os.path.join(self.target_reward_demos_dir, filename), **save_data),
                saved_demos += 1

        if self.target_int_demos_dir:

            # Create target dir for reward agent_demos if necessary
            if not os.path.exists(self.target_int_demos_dir):
                os.makedirs(self.target_int_demos_dir, exist_ok=True)

            # Rank agent agent_demos according to episode reward
            intrinsic_ranking = np.flip(np.array([d["IntrinsicReward"] for d in self.intrinsic_demos]).argsort())

            # Save agent intrinsic agent_demos
            saved_demos = 0
            for demo_pos in intrinsic_ranking:

                if saved_demos == self.num_int_demos_to_save:
                    break

                filename = "intrinsic_demo_{}".format(saved_demos + 1)

                save_data = {
                    prl.OBS: np.array(self.intrinsic_demos[demo_pos][prl.OBS]).astype(self.demo_dtypes[prl.OBS]),
                    prl.REW: np.array(self.intrinsic_demos[demo_pos][prl.REW]).astype(self.demo_dtypes[prl.REW]),
                    prl.ACT: np.array(self.intrinsic_demos[demo_pos][prl.ACT]).astype(self.demo_dtypes[prl.ACT]),
                    "FrameSkip": self.frame_skip}

                if prl.EMBED in self.data.keys() and prl.EMBED in self.demos_data_fields and \
                        prl.EMBED in self.intrinsic_demos[demo_pos].keys():
                    save_data.update({prl.EMBED: np.array(self.intrinsic_demos[demo_pos][prl.EMBED])})

                np.savez(os.path.join(self.target_int_demos_dir, filename), **save_data)
                saved_demos += 1
