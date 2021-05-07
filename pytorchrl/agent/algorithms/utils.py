import itertools
import torch


def get_gradients(*nets, grads_to_cpu=False):
    """Gets gradients for all parameters in nets."""

    params = itertools.chain(*[net.parameters() for net in nets])

    grads = []
    for p in params:
        if grads_to_cpu:
            if p.grad is not None:
                grads.append(p.grad.data.cpu().numpy())
            else:
                grads.append(None)
        else:
            if p.grad is not None:
                grads.append(p.grad)

    return grads

def set_gradients(*nets, gradients, device):
    """Sets gradients as the gradient vaues for all parameters in nets."""

    params = itertools.chain(*[net.parameters() for net in nets])

    for g, p in zip(gradients, params):
        if g is not None:
            p.grad = torch.from_numpy(g).to(device)