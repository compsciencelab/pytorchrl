from pytorchrl.agent.actors.memory_networks.gru_net import GruNet
from pytorchrl.agent.actors.memory_networks.lstm_net import LstmNet
from pytorchrl.agent.actors.memory_networks.lstm_encoder_decoder_net import LSTMEncoderDecoder


def get_memory_network(name):
    """Returns model class from name."""
    if name is None:
        return None
    elif name == "GRU":
        return GruNet
    elif name == "LSTM":
        return LstmNet
    elif name == "LSTMEncoderDecoder":
        return LSTMEncoderDecoder
    else:
        raise ValueError("Specified memory net model not found!")
