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


###### KL DIVERGENCE ###################################################################################################


def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def gaussian_kl(mu1, mu2, cov1, cov2):
    """
    Decoupled KL between two multivariate gaussian distribution

    C_μ = KL(f(x|μi,Σi)||f(x|μ,Σi))
    C_Σ = KL(f(x|μi,Σi)||f(x|μi,Σ))

    Adapted from https://github.com/daisatojp/mpo/blob/master/mpo/mpo.py

    Parameters
    ----------
    mu1: torch.tensor
        Mean distribution 1 - (B, n).
    mu2: torch.tensor
        Mean distribution 2 - (B, n).
    cov1: torch.tensor
        Covariance matrix distribution 1 - (B, n, n).
    cov2:
        Covariance matrix distribution 2 - (B, n, n)

    Returns
    -------
    kl_mu: scalar
        Mean term of the KL.
    kl_sigma: scalar
        Covariance term of the KL.

    ref : https://stanford.edu/~jduchi/projects/general_notes.pdf page.13
    """

    n = cov2.size(-1)
    mu1 = mu1.unsqueeze(-1)  # (B, n, 1)
    mu2 = mu2.unsqueeze(-1)  # (B, n, 1)

    sigma1 = cov1 @ bt(cov1)  # (B, n, n)
    sigma2 = cov2 @ bt(cov2)  # (B, n, n)
    sigma1_det = sigma1.det()  # (B,)
    sigma2_det = sigma2.det()  # (B,)
    sigma1_inv = sigma1.inverse()  # (B, n, n)
    sigma2_inv = sigma2.inverse()  # (B, n, n)

    # determinant can be minus due to numerical calculation error
    # https://github.com/daisatojp/mpo/issues/11
    sigma1_det = torch.clamp_min(sigma1_det, 1e-6)
    sigma2_det = torch.clamp_min(sigma2_det, 1e-6)

    inner_mu = ((mu2 - mu1).transpose(-2, -1) @ sigma1_inv @ (mu2 - mu1)).squeeze()  # (B,)
    inner_sigma = torch.log(sigma1_det / sigma2_det) - n + btr(sigma2_inv @ sigma1_inv)  # (B,)
    kl_mu = 0.5 * torch.mean(inner_mu)
    kl_sigma = 0.5 * torch.mean(inner_sigma)

    return kl_mu, kl_sigma
