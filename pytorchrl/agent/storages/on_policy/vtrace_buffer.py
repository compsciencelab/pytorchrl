import torch
from .vanilla_on_policy_buffer import VanillaOnPolicyBuffer as B


class VTraceBuffer(B):
    """
    Storage class for On-Policy algorithms with off-policy correction method
    V-trace (https://arxiv.org/abs/1506.02438).

    Parameters
    ----------
    size : int
        Storage capacity along time axis.
    device: torch.device
        CPU or specific GPU where data tensors will be placed and class
        computations will take place. Should be the same device where the
        actor model is located.
    """

    # Accepted data fields. Inserting other fields will raise AssertionError
    on_policy_data_fields = ("obs", "obs2", "rhs", "act", "rew", "val", "logp", "done")

    def __init__(self, size, device=torch.device("cpu")):

        super(VTraceBuffer, self).__init__(
            size=size,
            device=device)

    def before_gradients(self, actor, algo):
        """
        Before updating actor policy model, compute returns and advantages.

        Parameters
        ----------
        actor : ActorCritic
            An actor class instance.
        algo : Algo
            An algorithm class instance.
        """
        with torch.no_grad():
            _ = actor.get_action(
                self.data["obs"][self.step - 1],
                self.data["rhs"][self.step - 1],
                self.data["done"][self.step - 1])
            next_value = actor.get_value(
                self.data["obs"][self.step - 1],
                self.data["rhs"][self.step - 1],
                self.data["done"][self.step - 1]
            )

        self.data["ret"][self.step] = next_value
        self.compute_returns(algo.gamma)
        self.compute_vtrace(actor, algo)

    def after_gradients(self, actor, algo, batch, info):
        """
        After updating actor policy model, make sure self.step is at 0.

        Parameters
        ----------
        actor : Actor class
            An actor class instance.
        algo : Algo class
            An algorithm class instance.
        batch : dict
            Data batch used to compute the gradients.
        info : dict
            Additional relevant info from gradient computation.
        """
        self.data["obs"][0].copy_(self.data["obs"][self.step - 1])
        self.data["rhs"][0].copy_(self.data["rhs"][self.step - 1])
        self.data["done"][0].copy_(self.data["done"][self.step - 1])

        if self.step != 0:
            self.step = 0

    @torch.no_grad()
    def get_updated_action_log_probs(self, actor, algo):
        """
        Computes new log probabilities of actions stored in `storage`
        according to current `actor` version. It also uses the current
        `actor` version to update the value predictions.

        Parameters
        ----------
        actor : ActorCritic
            An actor class instance.
        algo : Algo
            An algorithm class instance.

        Returns
        -------
        new_logp : torch.tensor
            New action log probabilities.
        """

        len, num_proc = self.data["act"].shape[0:2]

        # Create batches without shuffling data
        batches = self.generate_batches(
            algo.num_mini_batch, algo.mini_batch_size,
            num_epochs=1, recurrent_ac=actor.is_recurrent, shuffle=False)

        # Obtain new value and log probability predictions
        new_val = []
        new_logp = []
        for batch in batches:
            obs, rhs, act, done = batch["obs"], batch["rhs"], batch["act"], batch["done"]
            (logp, _, _) = actor.evaluate_actions(obs, rhs, done, act)
            val = actor.get_value(obs)
            new_val.append(val)
            new_logp.append(logp)

        # Concatenate results
        if actor.is_recurrent:
            new_val = [p.view(len, num_proc // algo.num_mini_batch, -1) for p in new_val]
            self.data["val"][:-1] = torch.cat(new_val, dim=1)
            new_logp = [p.view(len, num_proc // algo.num_mini_batch, -1) for p in new_logp]
            new_logp = torch.cat(new_logp, dim=1)
        else:
            self.data["val"][:-1] = torch.cat(new_val, dim=0).view(len, num_proc, 1)
            new_logp = torch.cat(new_logp, dim=0).view(len, num_proc, 1)

        return new_logp

    @torch.no_grad()
    def compute_vtrace(self, new_policy, algo, clip_rho_thres=1.0, clip_c_thres=1.0):
        """
        Computes V-trace target values and advantage predictions and stores them,
        along with the updated action log probabilities, in `storage`.

        Parameters
        ----------
        new_policy : Actor
            An actor class instance.
        algo : Algo
            An algorithm class instance.
        """

        l = self.step if self.step != 0 else self.max_size

        with torch.no_grad():
            _ = new_policy.get_action(
                self.data["obs"][self.step - 1],
                self.data["rhs"][self.step - 1],
                self.data["done"][self.step - 1])
            next_value = new_policy.get_value(self.data["obs"][self.step - 1])
        self.data["val"][self.step] = next_value

        new_action_log_probs = self.get_updated_action_log_probs(new_policy, algo)

        log_rhos = (new_action_log_probs - self.data["logp"])
        clipped_rhos = torch.clamp(torch.exp(log_rhos), max=clip_rho_thres)
        clipped_cs = torch.clamp(torch.exp(log_rhos), max=clip_c_thres)

        deltas = clipped_rhos * (
            self.data["ret"][:-1] + algo.gamma * self.data["val"][1:] - self.data["val"][:-1])

        acc = torch.zeros_like(self.data["val"][-1])
        result = []
        for i in reversed(range(l)):
            acc = deltas[i] + algo.gamma * clipped_cs[i] * acc * (1 - self.data["done"][i + 1])
            result.append(acc)

        result.reverse()
        result.append(torch.zeros_like(self.data["val"][-1]))
        vs_minus_v_xs = torch.stack(result)

        vs = torch.add(vs_minus_v_xs, self.data["val"])
        adv = clipped_rhos * (self.data["ret"][:-1] + algo.gamma * vs[1:] - self.data["val"][:-1])

        self.data["ret"] = vs
        self.data["logp"] = new_action_log_probs
        self.data["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)


