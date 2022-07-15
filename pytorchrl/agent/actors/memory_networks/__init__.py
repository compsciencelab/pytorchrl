from pytorchrl.agent.actors.memory_networks.gru_net import GruNet
from pytorchrl.agent.actors.memory_networks.seq2seq import Seq2Seq


def get_memory_network(name):
    """Returns model class from name."""
    if name == "GRU":
        return GruNet
    elif name == "Seq2Seq":
        return Seq2Seq
    else:
        raise ValueError("Specified model not found!")
