from .mlp import MLP
from .cnn import CNN
from .fixup_cnn import FixupCNN


def get_feature_extractor(name):
    """Returns model class from name."""
    if name == "MLP":
        return MLP
    elif name == "CNN":
        return CNN
    elif name == "Fixup":
        return FixupCNN
    else:
        raise ValueError("Specified model not found!")

