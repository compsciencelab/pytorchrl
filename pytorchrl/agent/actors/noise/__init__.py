from .gaussian import Normal
from .ou import OUNoise
from .nonoise import NoNoise

def get_noise(name):
    """Returns noise model class from name."""
    if name == "Gauss":
        return Normal
    elif name == "OU":
        return OUNoise
    elif name == "NoNoise":
        return  NoNoise
    else:
        raise ValueError("Specified Noise not found!")