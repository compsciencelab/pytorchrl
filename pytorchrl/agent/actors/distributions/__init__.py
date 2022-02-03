from .gaussian import DiagGaussian, DiagGaussianEnsemble
from .categorical import Categorical
from .squashed_gaussian import SquashedGaussian
from .deterministic import Deterministic, DeterministicMB


def get_dist(name):
    """Returns model class from name."""
    if name == "Categorical":
        return Categorical
    elif name == "Gaussian":
        return DiagGaussian
    elif name == "SquashedGaussian":
        return SquashedGaussian
    elif name == "Deterministic":
        return Deterministic
    elif name == "DeterministicMB":
        return DeterministicMB
    else:
        raise ValueError("Specified model not found!")