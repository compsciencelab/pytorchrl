"""Utils to map the REINVENT pre-trained network checkpoint to the pytorchrl network."""

import torch
from pytorchrl.envs.generative_chemistry.vocabulary import ReinventVocabulary, LibinventVocabulary

reinvent_weights_mapping = {
    "_embedding.weight": "policy_net.feature_extractor._embedding.weight",
    "_rnn.weight_ih_l0": "policy_net.memory_net._rnn.weight_ih_l0",
    "_rnn.weight_hh_l0": "policy_net.memory_net._rnn.weight_hh_l0",
    "_rnn.bias_ih_l0": "policy_net.memory_net._rnn.bias_ih_l0",
    "_rnn.bias_hh_l0": "policy_net.memory_net._rnn.bias_hh_l0",
    "_rnn.weight_ih_l1": "policy_net.memory_net._rnn.weight_ih_l1",
    "_rnn.weight_hh_l1": "policy_net.memory_net._rnn.weight_hh_l1",
    "_rnn.bias_ih_l1": "policy_net.memory_net._rnn.bias_ih_l1",
    "_rnn.bias_hh_l1": "policy_net.memory_net._rnn.bias_hh_l1",
    "_rnn.weight_ih_l2": "policy_net.memory_net._rnn.weight_ih_l2",
    "_rnn.weight_hh_l2": "policy_net.memory_net._rnn.weight_hh_l2",
    "_rnn.bias_ih_l2": "policy_net.memory_net._rnn.bias_ih_l2",
    "_rnn.bias_hh_l2": "policy_net.memory_net._rnn.bias_hh_l2",
    "_linear.weight": "policy_net.dist.linear.weight",
    "_linear.bias": "policy_net.dist.linear.bias",
}

libinvent_weights_mapping = {
    "_encoder._embedding.0.weight": "policy_net.memory_net._encoder._embedding.0.weight",
    "_encoder._rnn.weight_ih_l0": "policy_net.memory_net._encoder._rnn.weight_ih_l0",
    "_encoder._rnn.weight_hh_l0": "policy_net.memory_net._encoder._rnn.weight_hh_l0",
    "_encoder._rnn.bias_ih_l0": "policy_net.memory_net._encoder._rnn.bias_ih_l0",
    "_encoder._rnn.bias_hh_l0": "policy_net.memory_net._encoder._rnn.bias_hh_l0",
    "_encoder._rnn.weight_ih_l0_reverse": "policy_net.memory_net._encoder._rnn.weight_ih_l0_reverse",
    "_encoder._rnn.weight_hh_l0_reverse": "policy_net.memory_net._encoder._rnn.weight_hh_l0_reverse",
    "_encoder._rnn.bias_ih_l0_reverse": "policy_net.memory_net._encoder._rnn.bias_ih_l0_reverse",
    "_encoder._rnn.bias_hh_l0_reverse": "policy_net.memory_net._encoder._rnn.bias_hh_l0_reverse",
    "_encoder._rnn.weight_ih_l1": "policy_net.memory_net._encoder._rnn.weight_ih_l1",
    "_encoder._rnn.weight_hh_l1": "policy_net.memory_net._encoder._rnn.weight_hh_l1",
    "_encoder._rnn.bias_ih_l1": "policy_net.memory_net._encoder._rnn.bias_ih_l1",
    "_encoder._rnn.bias_hh_l1": "policy_net.memory_net._encoder._rnn.bias_hh_l1",
    "_encoder._rnn.weight_ih_l1_reverse": "policy_net.memory_net._encoder._rnn.weight_ih_l1_reverse",
    "_encoder._rnn.weight_hh_l1_reverse": "policy_net.memory_net._encoder._rnn.weight_hh_l1_reverse",
    "_encoder._rnn.bias_ih_l1_reverse": "policy_net.memory_net._encoder._rnn.bias_ih_l1_reverse",
    "_encoder._rnn.bias_hh_l1_reverse": "policy_net.memory_net._encoder._rnn.bias_hh_l1_reverse",
    "_encoder._rnn.weight_ih_l2": "policy_net.memory_net._encoder._rnn.weight_ih_l2",
    "_encoder._rnn.weight_hh_l2": "policy_net.memory_net._encoder._rnn.weight_hh_l2",
    "_encoder._rnn.bias_ih_l2": "policy_net.memory_net._encoder._rnn.bias_ih_l2",
    "_encoder._rnn.bias_hh_l2": "policy_net.memory_net._encoder._rnn.bias_hh_l2",
    "_encoder._rnn.weight_ih_l2_reverse": "policy_net.memory_net._encoder._rnn.weight_ih_l2_reverse",
    "_encoder._rnn.weight_hh_l2_reverse": "policy_net.memory_net._encoder._rnn.weight_hh_l2_reverse",
    "_encoder._rnn.bias_ih_l2_reverse": "policy_net.memory_net._encoder._rnn.bias_ih_l2_reverse",
    "_encoder._rnn.bias_hh_l2_reverse": "policy_net.memory_net._encoder._rnn.bias_hh_l2_reverse",
    "_decoder._embedding.0.weight": "policy_net.memory_net._decoder._embedding.0.weight",
    "_decoder._rnn.weight_ih_l0": "policy_net.memory_net._decoder._rnn.weight_ih_l0",
    "_decoder._rnn.weight_hh_l0": "policy_net.memory_net._decoder._rnn.weight_hh_l0",
    "_decoder._rnn.bias_ih_l0": "policy_net.memory_net._decoder._rnn.bias_ih_l0",
    "_decoder._rnn.bias_hh_l0": "policy_net.memory_net._decoder._rnn.bias_hh_l0",
    "_decoder._rnn.weight_ih_l1": "policy_net.memory_net._decoder._rnn.weight_ih_l1",
    "_decoder._rnn.weight_hh_l1": "policy_net.memory_net._decoder._rnn.weight_hh_l1",
    "_decoder._rnn.bias_ih_l1": "policy_net.memory_net._decoder._rnn.bias_ih_l1",
    "_decoder._rnn.bias_hh_l1": "policy_net.memory_net._decoder._rnn.bias_hh_l1",
    "_decoder._rnn.weight_ih_l2": "policy_net.memory_net._decoder._rnn.weight_ih_l2",
    "_decoder._rnn.weight_hh_l2": "policy_net.memory_net._decoder._rnn.weight_hh_l2",
    "_decoder._rnn.bias_ih_l2": "policy_net.memory_net._decoder._rnn.bias_ih_l2",
    "_decoder._rnn.bias_hh_l2": "policy_net.memory_net._decoder._rnn.bias_hh_l2",
    "_decoder._attention._attention_linear.0.weight": "policy_net.memory_net._decoder._attention._attention_linear.0.weight",
    "_decoder._attention._attention_linear.0.bias": "policy_net.memory_net._decoder._attention._attention_linear.0.bias",
    "_decoder._linear.weight": "policy_net.dist.linear.weight",
    "_decoder._linear.bias": "policy_net.dist.linear.bias",
}


def adapt_reinvent_checkpoint(file_path):
    """Loads a Reinvent pretrained model and make the necessary changes for it to be compatible with pytorchrl"""

    if torch.cuda.is_available():
        save_dict = torch.load(file_path)
    else:
        save_dict = torch.load(file_path, map_location=lambda storage, loc: storage)

    # Change network weight names
    new_save_dict = {}
    for k in save_dict["network"].keys():
        new_save_dict[reinvent_weights_mapping[k]] = save_dict["network"][k]

    # Temporarily save network weight to /tmp/network_params
    torch.save(new_save_dict, "/tmp/network_params.tmp")

    # Remove unnecessary network parameters
    network_params = save_dict["network_params"]
    network_params.pop("cell_type", None)
    network_params.pop("embedding_layer_size", None)

    return ReinventVocabulary(save_dict["vocabulary"], save_dict["tokenizer"]), \
           save_dict["max_sequence_length"], save_dict["network_params"], "/tmp/network_params.tmp"


def adapt_libinvent_checkpoint(file_path):
    """Loads a Libinvent pretrained model and make the necessary changes for it to be compatible with pytorchrl"""

    if torch.cuda.is_available():
        save_dict = torch.load(file_path)
    else:
        save_dict = torch.load(file_path, map_location=lambda storage, loc: storage)

    # Change network weight names
    new_save_dict = {}
    for k in save_dict["decorator"]["state"].keys():
        new_save_dict[libinvent_weights_mapping[k]] = save_dict["decorator"]["state"][k]

    # Temporarily save network weight to /tmp/network_params
    torch.save(new_save_dict, "/tmp/network_params.tmp")

    vocabulary = LibinventVocabulary(
        save_dict["model"]["vocabulary"].scaffold_vocabulary,
        save_dict["model"]["vocabulary"].scaffold_tokenizer,
        save_dict["model"]["vocabulary"].decoration_vocabulary,
        save_dict["model"]["vocabulary"].decoration_tokenizer,
    )

    return vocabulary, save_dict["model"]["max_sequence_length"], \
           save_dict["decorator"]["params"], "/tmp/network_params.tmp"
