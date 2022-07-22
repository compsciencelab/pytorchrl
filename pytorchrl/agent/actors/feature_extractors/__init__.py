import gym
from pytorchrl.agent.actors.feature_extractors.mlp import MLP
from pytorchrl.agent.actors.feature_extractors.cnn import CNN
from pytorchrl.agent.actors.feature_extractors.dictnet import DictNet
from pytorchrl.agent.actors.feature_extractors.fixup_cnn import FixupCNN


def get_feature_extractor(name):
    """Returns model class from name."""
    if name == "MLP":
        return MLP
    elif name == "CNN":
        return CNN
    elif name == "Fixup":
        return FixupCNN
    elif name == "DictNet":
        return DictNet
    else:
        raise ValueError("Specified model not found!")


def default_feature_extractor(input_space):
    """
    Returns the default net to use as a feature extractor
    given input_space.

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    """

    if isinstance(input_space, gym.spaces.Dict):
        net = DictNet
    elif len(input_space.shape) <= 2:
        net = MLP
    elif len(input_space.shape) == 3:
        net = CNN
    else:
        raise NotImplementedError

    return net
