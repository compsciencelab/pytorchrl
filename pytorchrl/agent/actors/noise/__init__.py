from .gaussian import Normal
from .ou import OUNoise

def get_noise(name):
    """Returns noise model class from name."""
    if name == "Gauss":
        return Normal
    elif name == "OU":
        return OUNoise
    else:
        raise ValueError("Specified Noise not found!")