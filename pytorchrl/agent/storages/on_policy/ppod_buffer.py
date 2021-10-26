import os
import glob
import copy
import uuid
import torch
import numpy as np
from collections import defaultdict

import pytorchrl as prl
from pytorchrl.agent.storages.on_policy.gae_buffer import GAEBuffer as B


class PPODBuffer(B):
    """
    Storage class for PPO+D algorithm.

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
        Algorithm class instance
    envs: VecEnv
        Train environments vector.
    frame_skip : int
        Environment skips every `frame_skip`-th observation.
    frame_stack : int
        Environment observations composed of last `frame_stack` frames stacked.
    initial_demos_dir : str
        Path to directory containing initial demonstrations.
    target_demos_dir : str
        Path to directory where best demonstrations should be saved.
    rho : float
        PPO+D rho parameter.
    phi : float
        PPO+D phi parameter.
    alpha : float
        PPO+D alpha parameter
    gae_lambda : float
        GAE lambda parameter.
    max_demos : int
        Maximum number of demos to keep between reward and value demos.
    save_demo_frequency : int
        Save top demos every  `save_demo_frequency`th data collection.
    num_saved_demos : int
        Number of top reward demos to save.
    use_initial_demos_as_reward_threshold : bool
        Whether or not to use initially provided demos to define the reward threshold
        to accept future demos found by the agent.
    """

    # Accepted data fields. Inserting other fields will raise AssertionError
    on_policy_data_fields = prl.OnPolicyDataKeys

    # Data tensors to collect for each demos
    demos_data_fields = prl.DemosDataKeys

    def __init__(self, size, device, actor, algorithm, envs, frame_stack=1, frame_skip=0,
                 initial_demos_dir=None, target_demos_dir=None, rho=0.1, phi=0.3, gae_lambda=0.95,
                 alpha=10, max_demos=51, save_demo_frequency=10, num_saved_demos=10,
                 use_initial_demos_as_reward_threshold=True):

        super(PPODBuffer, self).__init__(
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
        self.initial_rho = rho
        self.initial_phi = phi
        self.max_demos = max_demos
        self.initial_demos_dir = initial_demos_dir
        self.target_demos_dir = target_demos_dir
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip

        # Reward and Value buffers
        self.reward_demos = []
        self.value_demos = []

        # Load initial demos
        if initial_demos_dir:
            self.load_initial_demos()
        
        # Define reward threshold for new demos 
        if use_initial_demos_as_reward_threshold:
            self.reward_threshold = min([d["TotalReward"] for d in self.reward_demos]) if len(
                self.reward_demos) > 0 else - np.inf
        else:
            self.reward_threshold = - np.inf

        # Define variables to track demos
        self.max_demo_reward = max([d["TotalReward"] for d in self.reward_demos]) if len(self.reward_demos) > 0 else 0.0
        self.potential_demos_val = {"env{}".format(i + 1): - np.inf for i in range(self.num_envs)}
        self.potential_demos = {"env{}".format(i + 1): defaultdict(list) for i in range(self.num_envs)}

        # Define variable to track demos in progress
        self.demos_in_progress = {
            "env{}".format(i + 1): {
                "ID": None,
                "Demo": None,
                "Step": 0,
                "DemoLength": -1,
                "MaxValue": - np.inf,
                prl.RHS: None,
            } for i in range(self.num_envs)}

        # Save demos
        self.iter = 0
        self.save_demos_every = save_demo_frequency
        self.num_saved_demos = num_saved_demos

    @classmethod
    def create_factory(cls,
                       size,
                       initial_demos_dir=None,
                       target_demos_dir=None,
                       frame_stack=1,
                       frame_skip=0,
                       rho=0.1,
                       phi=0.3,
                       gae_lambda=0.95,
                       alpha=10,
                       max_demos=51,
                       save_demo_frequency=10,
                       num_saved_demos=10,
                       use_initial_demos_as_reward_threshold=True):
        """
        Returns a function that creates PPODBuffer instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.
        initial_demos_dir : str
            Path to directory containing initial demonstrations.
        target_demos_dir : str
            Path to directory where best demonstrations should be saved.
        frame_skip : int
            Environment skips every `frame_skip`-th observation.
        frame_stack : int
            Environment observations composed of last `frame_stack` frames stacked.
        rho : float
            PPO+D rho parameter
        phi : float
            PPO+D phi parameter.
        alpha : float
            PPO+D alpha parameter
        gae_lambda : float
            GAE lambda parameter.
        max_demos : int
            Maximum number of demos to keep between reward and value demos.
        save_demo_frequency : int
            Save top demos every  `save_demo_frequency`th data collection.
        num_saved_demos : int
            Number of top reward demos to save.
        use_initial_demos_as_reward_threshold : bool
            Whether or not to use initially provided demos to define the reward threshold
            to accept future demos found by the agent.

        Returns
        -------
        create_buffer_instance : func
            creates a new PPODBuffer class instance.
        """

        def create_buffer_instance(device, actor, algorithm, envs):
            """Create and return a PPODBuffer instance."""
            return cls(size, device, actor, algorithm, envs, frame_stack,
                       frame_skip, initial_demos_dir, target_demos_dir, rho,
                       phi, gae_lambda, alpha, max_demos, save_demo_frequency,
                       num_saved_demos, use_initial_demos_as_reward_threshold)

        return create_buffer_instance

    def before_gradients(self):
        """
        Before updating actor policy model, compute returns and advantages.
        """

        print("\nREWARD DEMOS {}, VALUE DEMOS {}, RHO {}, PHI {}, REWARD THRESHOLD {}, MAX DEMO REWARD {}\n".format(
            len(self.reward_demos), len(self.value_demos), self.rho, self.phi, self.reward_threshold, self.max_demo_reward))

        # Retrieve most recent obs, rhs and done tensors
        last_tensors = {}
        step = self.step if self.step != 0 else -1
        for k in (prl.OBS, prl.RHS, prl.DONE):
            if isinstance(self.data[k], dict):
                last_tensors[k] = {x: self.data[k][x][step] for x in self.data[k]}
            else:
                last_tensors[k] = self.data[k][step]

        # Compute next value prediction
        with torch.no_grad():
            _ = self.actor.get_action(last_tensors[prl.OBS], last_tensors[prl.RHS], last_tensors[prl.DONE])
            value_dict = self.actor.get_value(last_tensors[prl.OBS], last_tensors[prl.RHS], last_tensors[prl.DONE])
            next_value = value_dict.get("value_net1")
            next_rhs = value_dict.get("rhs")

        # Assign predictions to self.data
        self.data[prl.VAL][step].copy_(next_value)
        if isinstance(next_rhs, dict):
            for x in self.data[prl.RHS]:
                self.data[prl.RHS][x][step].copy_(next_rhs[x])
        else:
            self.data[prl.RHS][step].copy_(next_rhs)

        self.compute_returns()
        self.compute_advantages()

        self.iter += 1
        if self.iter % self.save_demos_every == 0:
            self.save_demos()

    def get_num_channels_obs(self, sample):
        """
        Obtain num_channels_obs and set it as class attribute.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """
        self.num_channels_obs = int(sample[prl.OBS][0].shape[0] // self.frame_stack)

    def insert_transition(self, sample):
        """
        Store new transition sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """

        # Data tensors lazy initialization, only executed the first time
        if self.size == 0 and self.data[prl.OBS] is None:
            self.init_tensors(sample)
            self.get_num_channels_obs(sample)

        # Insert sample data
        for k in sample:

            if k not in self.storage_tensors:
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

        # Track episodes for potential demos
        self.track_potential_demos(sample)

        # Handle demos in progress
        for i in range(self.num_envs):

            # For each environment, insert demo transition if in the middle of a demos
            if self.demos_in_progress["env{}".format(i + 1)]["Demo"]:

                # Get next demo step to be inserted
                demo_step = self.demos_in_progress["env{}".format(i + 1)]["Step"]

                # Get demo obs, rhs and done tensors to run forward pass
                obs = self.data[prl.OBS][self.step][i].unsqueeze(0)
                if self.demos_in_progress["env{}".format(i + 1)][prl.RHS]:
                    rhs = self.demos_in_progress["env{}".format(i + 1)][prl.RHS]
                    done = torch.zeros(1, 1).to(self.device)
                else:
                    obs, rhs, done = self.actor.actor_initial_states(obs)

                # Run forward pass
                _, _, rhs2, algo_data = self.algo.acting_step(obs, rhs, done)

                # Insert demo act tensor to self.step
                self.data[prl.ACT][self.step][i].copy_(self.demos_in_progress["env{}".format(
                    i + 1)]["Demo"][prl.ACT][demo_step])

                # Insert demo rew tensor to self.step
                self.data[prl.REW][self.step][i].copy_(self.demos_in_progress["env{}".format(
                    i + 1)]["Demo"][prl.REW][demo_step])

                # Insert demo logprob to self.step. Demo action prob is 1.0, so logprob is 0.0
                self.data[prl.LOGP][self.step][i].copy_(torch.zeros(1))

                # Insert demo values predicted by the forward pass
                self.data[prl.VAL][self.step][i].copy_(algo_data[prl.VAL].squeeze())

                # Update demo_in_progress variables
                self.demos_in_progress["env{}".format(i + 1)]["Step"] += 1
                self.demos_in_progress["env{}".format(i + 1)][prl.RHS] = rhs2

                self.demos_in_progress["env{}".format(i + 1)]["MaxValue"] = max(
                    [algo_data[prl.VAL].item(), self.demos_in_progress["env{}".format(i + 1)]["MaxValue"]])

                # Handle end of demos
                if demo_step == self.demos_in_progress["env{}".format(i + 1)]["DemoLength"] - 1:

                    # If value demo
                    if "MaxValue" in self.demos_in_progress["env{}".format(i + 1)]["Demo"].keys():
                        for value_demo in self.value_demos:
                            # If demo still in buffer, update MaxValue
                            if self.demos_in_progress["env{}".format(i + 1)]["Demo"]["ID"] == value_demo["ID"]:
                                value_demo["MaxValue"] = self.demos_in_progress["env{}".format(i + 1)]["MaxValue"]

                    # Randomly sample new demos if last demos has finished
                    self.sample_demo(env_id=i)

                else:

                    # Insert demo done2 tensor to self.step + 1
                    self.data[prl.DONE][self.step + 1][i].copy_(torch.zeros(1))

                    # Insert demo obs2 tensor to self.step + 1
                    obs2 = torch.roll(obs, -self.num_channels_obs, dims=1).squeeze(0)
                    obs2[-self.num_channels_obs:].copy_(
                        self.demos_in_progress["env{}".format(i + 1)]["Demo"][prl.OBS][demo_step + 1].to(self.device))
                    self.data[prl.OBS][self.step + 1][i].copy_(obs2)

                    # Insert demo rhs2 tensor to self.step + 1
                    for k in self.data[prl.RHS]:
                        self.data[prl.RHS][k][self.step + 1][i].copy_(rhs2[k].squeeze())

            # Otherwise check if end of episode reached and randomly start new demo
            elif sample[prl.DONE2][i] == 1.0:

                self.sample_demo(env_id=i)

        self.step = (self.step + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def track_potential_demos(self, sample):
        """ Tracks current episodes looking for potential demos """

        for i in range(self.num_envs):

            # Copy transition
            # TODO. is deepcopy necessary?
            for tensor in self.demos_data_fields:
                if tensor in (prl.OBS):
                    self.potential_demos["env{}".format(i + 1)][tensor].append(
                        copy.deepcopy(sample[tensor][i, -self.num_channels_obs:]).cpu().numpy())
                else:
                    self.potential_demos["env{}".format(i + 1)][tensor].append(
                        copy.deepcopy(sample[tensor][i]).cpu().numpy())

            # Track highest value prediction
            self.potential_demos_val[i] = max([self.potential_demos_val["env{}".format(
                i + 1)], sample[prl.VAL][i].item()])

            # Handle end of episode
            if sample[prl.DONE2][i] == 1.0:

                # Get candidate demos
                potential_demo = {}
                for tensor in self.demos_data_fields:
                    potential_demo[tensor] = torch.Tensor(np.stack(
                        self.potential_demos["env{}".format(i + 1)][tensor]))

                # Compute accumulated reward
                episode_reward = potential_demo[prl.REW].sum().item()
                potential_demo["ID"] = str(uuid.uuid4())
                potential_demo["TotalReward"] = episode_reward
                potential_demo["DemoLength"] = potential_demo[prl.ACT].shape[0]

                # Consider candidate demos for demos reward
                if episode_reward >= self.reward_threshold:

                    # Add demos to reward buffer
                    self.reward_demos.append(potential_demo)

                    # Check if buffers are full
                    self.check_demo_buffer_capacity()

                    # Anneal rho and phi
                    self.anneal_parameters()

                    # Update reward_threshold.
                    self.reward_threshold = min([d["TotalReward"] for d in self.reward_demos])

                    # Also keep track of best demo reward
                    self.max_demo_reward = max([d["TotalReward"] for d in self.reward_demos])

                else:  # Consider candidate demos for value reward

                    # Find current number of demos, and current value threshold
                    potential_demo["MaxValue"] = self.potential_demos_val[i]
                    total_demos = len(self.reward_demos) + len(self.value_demos)
                    value_thresh = - np.float("Inf") if len(self.value_demos) == 0 \
                        else min([p["MaxValue"] for p in self.value_demos])

                    if self.potential_demos_val["env{}".format(i + 1)] >= value_thresh or total_demos < self.max_demos:

                        # Add demos to value buffer
                        self.value_demos.append(potential_demo)

                        # Check if buffers are full
                        self.check_demo_buffer_capacity()

                # Reset potential demos dict
                for tensor in self.demos_data_fields:
                    self.potential_demos["env{}".format(i + 1)][tensor] = []
                    self.potential_demos_val["env{}".format(i + 1)] = - np.inf

    def load_initial_demos(self):
        """
        Load initial demonstrations.

        Warning: make sure the frame_skip and frame_stack hyperparameters are
        the same as those used to record the demonstrations!
        """

        # Add original demonstrations
        num_loaded_demos = 0
        initial_demos = glob.glob(self.initial_demos_dir + '/*.npz')

        if len(initial_demos) > self.max_demos:
            raise ValueError("demo dir contains more than self.max_demos demonstrations")

        for demo_file in initial_demos:

            try:

                # Load demos tensors
                demo = np.load(demo_file)
                new_demo = {k: {} for k in self.demos_data_fields}

                if demo["FrameSkip"] != self.frame_skip:
                    raise ValueError(
                        "Env and demo with different frame skip!")

                # Add action
                demo_act = torch.FloatTensor(demo[prl.ACT])
                new_demo[prl.ACT] = demo_act

                # Add obs
                demo_obs = torch.FloatTensor(demo[prl.OBS])
                new_demo[prl.OBS] = demo_obs

                # Add rew
                demo_rew = torch.FloatTensor(demo[prl.REW])
                new_demo[prl.REW] = demo_rew

                new_demo.update({
                    "ID": str(uuid.uuid4()),
                    "DemoLength": demo[prl.ACT].shape[0],
                    "TotalReward": demo_rew.sum().item()})
                self.reward_demos.append(new_demo)
                num_loaded_demos += 1

            except Exception:
                print("Failed to load demo!")

        self.num_loaded_demos = num_loaded_demos
        print("\nLOADED {} DEMOS".format(num_loaded_demos))

    def sample_demo(self, env_id):
        """With probability rho insert reward demos, with probability phi insert value demos."""

        # Reset demos tracking variables
        self.demos_in_progress["env{}".format(env_id + 1)]["Step"] = 0
        self.demos_in_progress["env{}".format(env_id + 1)][prl.RHS] = None

        # Sample episode type
        episode_source = np.random.choice(["reward_demo", "value_demo", "env"],
            p=[self.rho, self.phi, 1.0 - self.rho - self.phi])

        if episode_source == "reward_demo" and len(self.reward_demos) > 0:

            # Randomly select a reward demo
            selected = np.random.choice(range(len(self.reward_demos)))

            # give priority to shorter demos
            # probs = 1 / np.array([p["obs"].shape[0] for p in self.reward_demos])
            # probs = probs / probs.sum()
            # selected = np.random.choice(range(len(self.reward_demos)), p=probs)

            demo = copy.deepcopy(self.reward_demos[selected])

        elif episode_source == "value_demo" and len(self.value_demos) > 0:

            # randomly select a value demo
            probs = np.array([p["MaxValue"] for p in self.value_demos]) ** self.alpha
            probs = probs / probs.sum()
            selected = np.random.choice(range(len(self.value_demos)), p=probs)
            demo = copy.deepcopy(self.value_demos[selected])

        else:
            demo = None

        # Set demos to demos_in_progress
        self.demos_in_progress["env{}".format(env_id + 1)]["Demo"] = demo

        # Set done to True
        self.data[prl.DONE][self.step + 1][env_id].copy_(torch.ones(1).to(self.device))

        # Set initial rhs to zeros
        for k in self.data[prl.RHS]:
            self.data[prl.RHS][k][self.step + 1][env_id].fill_(0.0)

        if demo:

            # Set demos length
            self.demos_in_progress["env{}".format(env_id + 1)]["DemoLength"] = demo["DemoLength"]

            # Set demos MaxValue
            self.demos_in_progress["env{}".format(env_id + 1)]["MaxValue"] = - np.Inf

            # Set next buffer obs to be the starting demo obs
            for k in range(self.frame_stack):
                self.data[prl.OBS][self.step + 1][env_id][
                k * self.num_channels_obs:(k + 1) * self.num_channels_obs].copy_(
                    self.demos_in_progress["env{}".format(env_id + 1)]["Demo"][prl.OBS][0].to(self.device))

        else:
            # Reset `i-th` environment as set next buffer obs to be the starting episode obs
            self.data[prl.OBS][self.step + 1][env_id].copy_(self.envs.reset_single_env(env_id=env_id).squeeze())

            # Reset potential demos dict
            for tensor in self.demos_data_fields:
                self.potential_demos["env{}".format(env_id + 1)][tensor] = []
                self.potential_demos_val["env{}".format(env_id + 1)] = - np.inf

    def anneal_parameters(self):
        """Update demos probabilities as explained in PPO+D paper."""

        if 0.0 < self.rho < 1.0 and len(self.value_demos) > 0:
            self.rho += self.initial_phi / len(self.value_demos)
            self.rho = np.clip(self.rho, 0.0, self.initial_rho + self.initial_phi)

        if 0.0 < self.phi < 1.0 and len(self.value_demos) > 0:
            self.phi -= self.initial_phi / len(self.value_demos)
            self.phi = np.clip(self.phi, 0.0, self.initial_rho + self.initial_phi)

    def check_demo_buffer_capacity(self):
        """
        Check total amount of demos. If total amount of demos exceeds
        self.max_demos, pop demos.
        """

        # First pop value demos
        total_demos = len(self.reward_demos) + len(self.value_demos)
        if total_demos > self.max_demos:
            for _ in range(min(total_demos - self.max_demos, len(self.value_demos))):
                # Pop value demos with lowest MaxValue
                del self.value_demos[np.array([p["MaxValue"] for p in self.value_demos]).argmin()]

        # If after popping all value demos, still over max_demos, pop reward demos
        if len(self.reward_demos) > self.max_demos:
            # Randomly remove reward demos, longer demos have higher probability
            for _ in range(len(self.reward_demos) - self.max_demos):

                # Option 1: FIFO (original paper)
                # del self.reward_demos[self.num_loaded_demos]

                # Option 2: Eject demo with lowest reward
                rewards = np.array([p[prl.REW].sum() for p in self.reward_demos[self.num_loaded_demos:]])
                del self.reward_demos[np.argmin(rewards) + self.num_loaded_demos]

    def save_demos(self):
        """
        Saves the top `num_rewards_demos` demos from the reward demos buffer and
        the top `num_value_demos` demos from the value demos buffer.
        """

        if self.target_demos_dir and not os.path.exists(self.target_demos_dir):
            os.makedirs(self.target_demos_dir, exist_ok=True)

        reward_ranking = np.flip(np.array(
            [d["TotalReward"] for d in self.reward_demos[
                self.num_loaded_demos:]]).argsort())[:self.num_saved_demos]

        for num, demo_pos in enumerate(reward_ranking):
            filename = os.path.join(self.target_demos_dir, "reward_demo_{}".format(num + 1))
            demo_pos += self.num_loaded_demos
            np.savez(
                filename,
                Observation=np.array(self.reward_demos[demo_pos][prl.OBS]).astype(np.float32),
                Reward=np.array(self.reward_demos[demo_pos][prl.REW]).astype(np.float32),
                Action=np.array(self.reward_demos[demo_pos][prl.ACT]).astype(np.float32),
            )
