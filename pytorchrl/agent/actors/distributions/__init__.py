from .gaussian import DiagGaussian, DiagGaussianEnsemble
from .categorical import Categorical
from .squashed_gaussian import SquashedGaussian
from .deterministic import Deterministic, DeterministicEnsemble


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
    elif name == "DeterministicEnsemble":
        return DeterministicEnsemble
    elif name == "DiagGaussianEnsemble":
        return DiagGaussianEnsemble
    else:
        raise ValueError("Specified model not found!")