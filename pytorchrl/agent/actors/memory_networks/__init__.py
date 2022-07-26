from pytorchrl.agent.actors.memory_networks.gru_net import GruNet
from pytorchrl.agent.actors.memory_networks.lstm_net import LstmNet


def get_memory_network(name):
    """Returns model class from name."""
    if name is None:
        return None
    elif name == "GRU":
        return GruNet
    elif name == "LSTM":
        return LstmNet
    else:
        raise ValueError("Specified memory net model not found!")
