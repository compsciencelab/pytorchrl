import torch
import pytorchrl as prl
from pytorchrl.agent.storages.on_policy.vanilla_on_policy_buffer import VanillaOnPolicyBuffer as B


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
    actor : ActorCritic
        An actor class instance.
    algorithm : Algo
        An algorithm class instance.
    """

    # Data fields to store in buffer and contained in generated batches
    storage_tensors = prl.OnPolicyDataKeys

    def __init__(self, size, device, actor, algorithm):

        super(VTraceBuffer, self).__init__(
            size=size,
            device=device,
            actor=actor,
            algorithm=algorithm)

    def before_gradients(self):
        """
        Before updating actor policy model, compute returns and advantages.
        """

        last_tensors = {}
        for k in (prl.OBS, prl.RHS, prl.DONE):
            if isinstance(self.data[k], dict):
                last_tensors[k] = {x: self.data[k][x][self.step - 1] for x in self.data[k]}
            else:
                last_tensors[k] = self.data[k][self.step - 1]

        with torch.no_grad():
            _ = self.actor.get_action(last_tensors[prl.OBS], last_tensors[prl.RHS], last_tensors[prl.DONE])
            next_value, next_rhs = self.actor.get_value(last_tensors[prl.OBS], last_tensors[prl.RHS], last_tensors[prl.DONE])

        self.data[prl.RET][self.step].copy_(next_value)
        self.data[prl.VAL][self.step].copy_(next_value)

        if isinstance(next_rhs, dict):
            for x in self.data[prl.RHS]:
                self.data[prl.RHS][x][self.step].copy_(next_rhs[x])
        else:
            self.data[prl.RHS][self.step] = next_rhs
        self.compute_returns()

        self.compute_vtrace()

    @torch.no_grad()
    def get_updated_action_log_probs(self):
        """
        Computes new log probabilities of actions stored in `storage`
        according to current `actor` version. It also uses the current
        `actor` version to update the value predictions.
        """

        l, num_proc = self.data[prl.DONE].shape[0:2]
        l = self.step if self.step != 0 else self.max_size

        # Create batches without shuffling data
        batches = self.generate_batches(
            self.algo.num_mini_batch, self.algo.mini_batch_size,
            num_epochs=1, shuffle=False)

        # Obtain new value and log probability predictions
        new_val = []
        new_logp = []
        for batch in batches:
            obs, rhs, act, done = batch[prl.OBS], batch[prl.RHS], batch[prl.ACT], batch[prl.DONE]
            (logp, _, _) = self.actor.evaluate_actions(obs, rhs, done, act)
            val, _ = self.actor.get_value(obs, rhs, done)
            new_val.append(val)
            new_logp.append(logp)

        # Concatenate results
        if self.actor.is_recurrent:
            new_val = [p.view(l, num_proc // self.algo.num_mini_batch, -1) for p in new_val]
            self.data[prl.VAL][:-1] = torch.cat(new_val, dim=1)
            new_logp = [p.view(l, num_proc // self.algo.num_mini_batch, -1) for p in new_logp]
            new_logp = torch.cat(new_logp, dim=1)
        else:
            self.data[prl.VAL][:-1] = torch.cat(new_val, dim=0).view(l, num_proc, 1)
            new_logp = torch.cat(new_logp, dim=0).view(l, num_proc, 1)

        return new_logp

    @torch.no_grad()
    def compute_vtrace(self, clip_rho_thres=1.0, clip_c_thres=1.0):
        """
        Computes V-trace target values and advantage predictions and stores them,
        along with the updated action log probabilities, in `storage`.

        Parameters
        ----------
        clip_rho_thres : float
            V-trace rho threshold parameter.
        clip_c_thres : float
            V-trace c threshold parameter.
        """

        l = self.step if self.step != 0 else self.max_size

        new_action_log_probs = self.get_updated_action_log_probs(self.actor, self.algo)

        log_rhos = (new_action_log_probs - self.data[prl.LOGP][:l])
        clipped_rhos = torch.clamp(torch.exp(log_rhos), max=clip_rho_thres)
        clipped_cs = torch.clamp(torch.exp(log_rhos), max=clip_c_thres)

        deltas = clipped_rhos * (
            self.data[prl.RET][:-1] + self.algo.gamma * self.data[prl.VAL][1:]
            - self.data[prl.VAL][:-1])

        acc = torch.zeros_like(self.data[prl.VAL][-1])
        result = []
        for i in reversed(range(l)):
            acc = deltas[i] + self.algo.gamma * clipped_cs[i] * acc * (
                1 - self.data[prl.DONE][i + 1])
            result.append(acc)

        result.reverse()
        result.append(torch.zeros_like(self.data[prl.VAL][-1]))
        vs_minus_v_xs = torch.stack(result)

        vs = torch.add(vs_minus_v_xs, self.data[prl.VAL])
        adv = clipped_rhos * (self.data[prl.RET][:-1] + self.algo.gamma *
                              vs[1:] - self.data[prl.VAL][:-1])

        self.data[prl.RET] = vs
        self.data[prl.LOGP] = new_action_log_probs
        self.data[prl.ADV] = (adv - adv.mean()) / (adv.std() + 1e-8)
