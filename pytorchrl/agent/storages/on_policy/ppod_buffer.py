import glob
import copy
import torch
import numpy as np
from collections import defaultdict

import pytorchrl as prl
from pytorchrl.agent.storages.on_policy.gae_buffer import GAEBuffer as B


class PPODBuffer(B):
    """
    Storage class for PPO+D algorithm. To minimize limitations make buffer
    length large enough wrt max episode size (ideally, size > max_demo_length).

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
    """

    # Accepted data fields. Inserting other fields will raise AssertionError
    on_policy_data_fields = prl.OnPolicyDataKeys

    # Data tensors to collect for each demo
    demos_data_fields = prl.DemosDataKeys

    def __init__(self, size, device, actor, algorithm, envs, demos_dir=None, rho=0.1, phi=0.0, gae_lambda=0.95, alpha=10, max_demos=51):

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
        self.original_demos_dir = demos_dir

        # Reward and Value buffers
        self.reward_demos = []
        self.value_demos = []

        # Track potential demos
        self.potential_demos_val = defaultdict(float)
        self.potential_demos = {"env{}".format(i + 1): defaultdict(list) for i in range(self.num_envs)}

        if demos_dir:
            self.load_original_demos()
            self.reward_threshold = min([d["total_reward"] for d in self.reward_demos])
        else:
            self.reward_threshold = - np.inf

        # Track demos in progress
        self.demos_in_progress = {"env{}".format(i + 1): {
            "Demo": None,
            "Step": 0,
            "DemoLength": -1,
            prl.RHS: None,
        } for i in range(self.num_envs)}

    @classmethod
    def create_factory(cls, size, demos_dir=None, rho=0.1, phi=0.0, gae_lambda=0.95, alpha=10, max_demos=51):
        """
        Returns a function that creates PPODBuffer instances.
        Parameters
        ----------
        size : int
            Storage capacity along time axis.
        gae_lambda : float
            GAE lambda parameter.
        Returns
        -------
        create_buffer_instance : func
            creates a new PPODBuffer class instance.
        """

        def create_buffer_instance(device, actor, algorithm, envs):
            """Create and return a PPODBuffer instance."""
            return cls(size, device, actor, algorithm, envs, demos_dir, rho, phi, gae_lambda, alpha, max_demos)

        return create_buffer_instance

    def before_gradients(self):
        """
        Before updating actor policy model, compute returns and advantages.
        """

        print("\nREWARD DEMOS {}, VALUE DEMOS {}, RHO {}, PHI {}".format(
            len(self.reward_demos), len(self.value_demos), self.rho, self.phi))

        print("\nREWARD THRESHOLD {}".format(self.reward_threshold))

        last_tensors = {}
        for k in (prl.OBS, prl.RHS, prl.DONE):
            if isinstance(self.data[k], dict):
                last_tensors[k] = {x: self.data[k][x][self.step - 1] for x in self.data[k]}
            else:
                last_tensors[k] = self.data[k][self.step - 1]

        with torch.no_grad():
            _ = self.actor.get_action(last_tensors[prl.OBS], last_tensors[prl.RHS], last_tensors[prl.DONE])
            value_dict = self.actor.get_value(last_tensors[prl.OBS], last_tensors[prl.RHS], last_tensors[prl.DONE])
            next_value = value_dict.get("value_net1")
            next_rhs = value_dict.get("rhs")

        self.data[prl.RET][self.step].copy_(next_value)
        self.data[prl.VAL][self.step].copy_(next_value)

        if isinstance(next_rhs, dict):
            for x in self.data[prl.RHS]:
                self.data[prl.RHS][x][self.step].copy_(next_rhs[x])
        else:
            self.data[prl.RHS][self.step] = next_rhs

        self.compute_returns()
        self.compute_advantages()

    def insert_transition(self, sample):
        """
        Store new transition sample.
        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """

        # Data tensors lazy initialization
        if self.size == 0 and self.data[prl.OBS] is None:
            self.init_tensors(sample)

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

        # For each environment, replace sample data with demo data if demo in progress
        for i in range(self.num_envs):

            # Insert demo step if in the middle of a demo
            if self.demos_in_progress["env{}".format(i + 1)]["Demo"]:

                demo_step = self.demos_in_progress["env{}".format(i + 1)]["Step"]
                obs = self.demos_in_progress["env{}".format(i + 1)]["Demo"][prl.OBS][demo_step:demo_step + 1].to(self.device)  # get last obs
                obs2 = self.demos_in_progress["env{}".format(i + 1)]["Demo"][prl.OBS][demo_step + 1:demo_step + 2].to(self.device)  # get last obs
                if self.demos_in_progress["env{}".format(i + 1)][prl.RHS]:
                    rhs = self.demos_in_progress["env{}".format(i + 1)][prl.RHS]
                    done = torch.zeros(1, 1).to(self.device)
                else:
                    obs, rhs, done = self.actor.actor_initial_states(obs)

                _, _, rhs2, algo_data = self.algo.acting_step(obs, rhs, done)

                # Copy tensors to self.data
                self.data[prl.OBS][self.step][i:i + 1, :].copy_(obs2)
                self.data[prl.DONE][self.step][i:i + 1, :].copy_(done)
                self.data[prl.ACT][self.step][i:i + 1, :].copy_(self.demos_in_progress["env{}".format(i + 1)]["Demo"][prl.ACT][demo_step])
                self.data[prl.REW][self.step][i:i + 1, :].copy_(self.demos_in_progress["env{}".format(i + 1)]["Demo"][prl.REW][demo_step])
                self.data[prl.LOGP][self.step][i:i + 1, :].copy_(torch.zeros(1, 1))
                self.data[prl.VAL][self.step][i:i + 1, :].copy_(algo_data[prl.VAL])
                for k in self.data[prl.RHS]:
                    self.data[prl.RHS][k][self.step][i:i + 1, :].copy_(rhs2[k])
                self.demos_in_progress["env{}".format(i + 1)][prl.RHS] = rhs2

                self.demos_in_progress["env{}".format(i + 1)]["Step"] += 1
                if self.demos_in_progress["env{}".format(i + 1)]["Step"] == self.demos_in_progress["env{}".format(i + 1)]["DemoLength"] - 1:
                    self.data[prl.DONE][self.step][i:i + 1, :].copy_(torch.ones(1, 1).to(self.device))
                    self.data[prl.OBS][self.step][i:i + 1, :].copy_(self.envs.reset_single_env(env_id=i))  # TODO. make sure it is obs in next transition

                    # Here start demo if done last episode and prob says a demo goes now
                    self.demos_in_progress["env{}".format(i + 1)]["Step"] = 0
                    self.demos_in_progress["env{}".format(i + 1)]["Demo"] = self.sample_demo()
                    if self.demos_in_progress["env{}".format(i + 1)]["Demo"]:
                        self.demos_in_progress["env{}".format(i + 1)]["DemoLength"] = \
                            self.demos_in_progress["env{}".format(i + 1)]["Demo"][prl.OBS].shape[0]

            # Otherwise check if end of episode reach and randomly start demo
            elif sample[prl.DONE][i] == 1.0:

                self.demos_in_progress["env{}".format(i + 1)]["Step"] = 0
                self.demos_in_progress["env{}".format(i + 1)]["Demo"] = self.sample_demo()
                if self.demos_in_progress["env{}".format(i + 1)]["Demo"]:
                    self.demos_in_progress["env{}".format(i + 1)]["DemoLength"] = \
                        self.demos_in_progress["env{}".format(i + 1)]["Demo"][prl.OBS].shape[0]

        self.step = (self.step + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def track_potential_demos(self, sample):
        """ Tracks current episodes looking for potential demos """

        for i in range(self.num_envs):

            for tensor in self.demos_data_fields + (prl.VAL, ):
                self.potential_demos["env{}".format(i + 1)][tensor].append(sample[tensor][i].cpu().numpy())

            # TODO. is this correct ?
            self.potential_demos_val[i] = max([self.potential_demos_val["env{}".format(i + 1)], sample[prl.VAL][i].item()])

            if sample[prl.DONE][i] == 1.0:
                potential_demo = {}
                for tensor in self.demos_data_fields:
                    potential_demo[tensor] = torch.Tensor(np.stack(self.potential_demos["env{}".format(i + 1)][tensor]))

                # Compute accumulated reward
                episode_reward = potential_demo[prl.REW].sum().item()
                potential_demo["total_reward"] = episode_reward
                potential_demo["length"] = potential_demo[prl.ACT].shape[0]

                if episode_reward > self.reward_threshold:

                    # Add demo to reward buffer
                    self.reward_demos.append(potential_demo)

                    # Check if buffers are full
                    self.check_demo_buffer_capacity()

                    # Anneal rho and phi
                    self.anneal_parameters()

                    # Update reward_threshold. TODO. review, this is not in the original paper.
                    self.reward_threshold = min([d["total_reward"] for d in self.reward_demos])

                else:

                    # Find current number of demos, and current minimum max value
                    potential_demo["max_value"] = torch.Tensor(np.stack(self.potential_demos["env{}".format(i + 1)][prl.VAL])).max().item()
                    total_demos = len(self.reward_demos) + len(self.value_demos)
                    value_thresh = - np.float("Inf") if len(self.value_demos) == 0 \
                        else min([p["max_value"] for p in self.value_demos])

                    if self.potential_demos_val["env{}".format(i + 1)] > value_thresh or total_demos < self.max_demos:

                        # Add demo to value buffer
                        self.value_demos.append(potential_demo)

                        # Check if buffers are full
                        self.check_demo_buffer_capacity()

                for tensor in self.demos_data_fields:
                    self.potential_demos["env{}".format(i + 1)][tensor] = []
                    self.potential_demos_val["env{}".format(i + 1)] = 0.0

    def load_original_demos(self):
        """Load initial demonstrations."""

        # Add original demonstrations
        original_demos = glob.glob(self.original_demos_dir + '/*.npz')
        for demo_file in original_demos:

            # Load demo tensors
            demo = np.load(demo_file)
            new_demo = {k: {} for k in self.demos_data_fields}

            # Add action
            demo_act = torch.FloatTensor(demo[prl.ACT])
            new_demo[prl.ACT] = demo_act

            # Add obs
            demo_obs = torch.FloatTensor(demo[prl.OBS])
            new_demo[prl.OBS] = demo_obs

            # Add rew, define success reward threshold
            demo_rew = torch.FloatTensor(demo[prl.OBS])
            new_demo[prl.REW] = demo_rew

            new_demo.update({"length": demo[prl.ACT].shape[0], "total_reward": demo_rew.sum().item()})

    def sample_demo(self):
        """With probability rho insert reward demo, with probability phi insert value demo."""

        # Sample episode type
        episode_source = np.random.choice(["reward_demo", "value_demo", "env"],
            p=[self.rho, self.phi, 1.0 - self.rho - self.phi])

        if episode_source == "reward_demo" and len(self.reward_demos) > 0:

            # Randomly select reward demo
            selected = np.random.choice(range(len(self.reward_demos)))

            # give priority to shorter demos
            # probs = 1 / np.array([p["obs"].shape[0] for p in self.reward_demos])
            # probs = probs / probs.sum()
            # selected = np.random.choice(range(len(self.reward_demos)), p=probs)

            demo = self.reward_demos[selected]

        elif episode_source == "value_demo" and len(self.value_demos) > 0:

            # randomly select value demo
            probs = np.array([p["max_value"] for p in self.value_demos]) ** self.alpha
            probs = probs / probs.sum()
            selected = np.random.choice(range(len(self.value_demos)), p=probs)
            demo = self.value_demos[selected]

        else:
            demo = None

        return copy.deepcopy(demo)

    def anneal_parameters(self):
        """Update demo probabilities as explained in PPO+D paper."""

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
                # pop value demo with lowest max_value
                del self.value_demos[np.array(
                    [p["max_value"] for p in self.value_demos]).argmin()]

        # If after popping all value demos, still over max_demos, pop reward demos
        if len(self.reward_demos) > self.max_demos:
            # Randomly remove reward demos, longer demos have higher probability
            for _ in range(len(self.reward_demos) - self.max_demos):
                probs = np.array([p[prl.OBS].shape[0] for p in self.reward_demos])
                probs = probs / probs.sum()
                del self.reward_demos[np.random.choice(range(len(self.reward_demos)), p=probs)]
