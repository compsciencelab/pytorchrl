from .gaussian import DiagGaussian
from .categorical import Categorical
from .squashed_gaussian import SquashedGaussian
from .deterministic import Deterministic


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
    else:
        raise ValueError("Specified model not found!")