from .gaussian import DiagGaussian
from .categorical import Categorical
from .squashed_gaussian import SquashedGaussian


def get_dist(name):
    """Returns model class from name."""
    if name == "Categorical":
        return Categorical
    elif name == "Gaussian":
        return DiagGaussian
    elif name == "SquashedGaussian":
        return SquashedGaussian
    else:
        raise ValueError("Specified model not found!")