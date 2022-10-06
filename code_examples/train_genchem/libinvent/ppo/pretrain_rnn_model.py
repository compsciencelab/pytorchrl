"""
Pretrain a GRU or LSTM model.
Code Adapted from https://github.com/MolecularAI/Lib-INVENT to work on PyTorchRL.
Requires preprocessed data as explained in https://github.com/MolecularAI/Lib-INVENT-dataset.
"""

import os
import glob
import wandb
import argparse
from tqdm import tqdm
import itertools as it
import torch
import torch.nn.utils.rnn as tnnur
from torch.utils.data import Dataset, DataLoader
from reinvent_chemistry.file_reader import FileReader

import pytorchrl as prl
from pytorchrl.agent.env import VecEnv
from pytorchrl.utils import LoadFromFile, save_argparse, cleanup_log_dir
from pytorchrl.envs.generative_chemistry.vocabulary import LibinventVocabulary
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor, get_memory_network
from pytorchrl.envs.generative_chemistry.libinvent.generative_chemistry_env_factory import libinvent_train_env_factory
from code_examples.train_genchem.libinvent.ppo.train_rnn_model import get_args


def load_dataset(path):
    reader = FileReader([], None)
    return list(reader.read_library_design_data_file(path, num_fields=2))


class DecoratorDataset(Dataset):
    """Dataset that takes a list of (scaffold, decoration) pairs."""

    def __init__(self, scaffold_decoration_smi_list, vocabulary):
        self.vocabulary = vocabulary

        self._encoded_list = []
        for scaffold, dec in tqdm(scaffold_decoration_smi_list):
            en_scaff = self.vocabulary.scaffold_vocabulary.encode(self.vocabulary.scaffold_tokenizer.tokenize(scaffold))
            en_dec = self.vocabulary.decoration_vocabulary.encode(self.vocabulary.decoration_tokenizer.tokenize(dec))
            if en_scaff is not None and en_dec is not None:
                self._encoded_list.append((en_scaff, en_dec))

    def __getitem__(self, i):
        scaff, dec = self._encoded_list[i]
        return torch.tensor(scaff, dtype=torch.long), torch.tensor(dec, dtype=torch.long)

    def __len__(self):
        return len(self._encoded_list)

    @classmethod
    def collate_fn(cls, encoded_pairs):
        """
        Turns a list of encoded pairs (scaffold, decoration) of sequences and turns them into two batches.
        :param: A list of pairs of encoded sequences.
        :return: A tuple with two tensors, one for the scaffolds and one for the decorations in the same order as given.
        """
        encoded_scaffolds, encoded_decorations = list(zip(*encoded_pairs))
        return pad_batch(encoded_scaffolds), pad_batch(encoded_decorations)


def pad_batch(encoded_seqs):
    """
    Pads a batch.
    :param encoded_seqs: A list of encoded sequences.
    :return: A tensor with the sequences correctly padded.
    """
    seq_lengths = torch.tensor([len(seq) for seq in encoded_seqs], dtype=torch.int64)
    return tnnur.pad_sequence(encoded_seqs, batch_first=True), seq_lengths


if __name__ == "__main__":

    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)
    save_argparse(args, os.path.join(args.log_dir, "conf.yaml"), [])
    pretrained_ckpt = {}
    os.makedirs("data", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load training set
    print("\nLoading data...")
    if not os.path.exists(args.pretrainingset_path):
        raise ValueError(f"Missing training set: {args.pretrainingset_path}")
    training_set = load_dataset(args.pretrainingset_path)

    # Create or load vocabularies
    if not os.path.exists(f"{args.log_dir}/pretrained_ckpt.prior"):
        print("\nConstructing vocabularies...")
        scaffold_list = [i[0] for i in training_set]
        decoration_list = [i[1] for i in training_set]
        vocabulary = LibinventVocabulary.from_lists(scaffold_list, decoration_list)  # Takes a long time!
        pretrained_ckpt["vocabulary"] = vocabulary
        pretrained_ckpt["max_sequence_length"] = args.pretrain_max_smile_length
        torch.save(pretrained_ckpt, f"{args.log_dir}/pretrained_ckpt.prior")
    else:
        print(f"\nCheckpoint {args.log_dir}/pretrained_ckpt.prior found. Loading...")
        pretrained_ckpt = torch.load(f"{args.log_dir}/pretrained_ckpt.prior")
        vocabulary = pretrained_ckpt["vocabulary"]
        pretrained_ckpt["max_sequence_length"] = args.pretrain_max_smile_length

    # Handle wandb init
    if args.wandb_key:
        mode = "online"
        wandb.login(key=str(args.wandb_key))
    else:
        mode = "disabled"

    with wandb.init(project=args.experiment_name, name=args.agent_name + "_pretrain", config=args, mode=mode):

        print("\nPreparing dataset and dataloader...")
        dataset = DecoratorDataset(training_set, vocabulary=vocabulary)  # Takes a long time!
        data = DataLoader(
            dataset,
            batch_size=args.pretrain_batch_size,
            shuffle=True, drop_last=True,
            collate_fn=dataset.collate_fn)

        # Define env
        test_env, action_space, obs_space = VecEnv.create_factory(
            env_fn=libinvent_train_env_factory,
            env_kwargs={
                "scoring_function": lambda a: {"reward": 1.0},
                "vocabulary": vocabulary, "smiles_max_length": args.pretrain_max_smile_length,
                "scaffolds": ["[*]"]
            },
            vec_env_size=1)
        env = test_env(device)

        # Define model
        feature_extractor_kwargs = {}
        recurrent_net_kwargs = {
            "encoder_params": {"num_layers": 3, "num_dimensions": 512, "vocabulary_size": 38, "dropout": 0.2},
            "decoder_params": {"num_layers": 3, "num_dimensions": 512, "vocabulary_size": 36, "dropout": 0.2}
        }
        actor = OnPolicyActor.create_factory(
            obs_space, action_space, prl.PPO,
            feature_extractor_network=torch.nn.Identity,
            feature_extractor_kwargs=feature_extractor_kwargs,
            recurrent_net=get_memory_network(args.recurrent_net),
            recurrent_net_kwargs={**recurrent_net_kwargs})(device)
        pretrained_ckpt["feature_extractor_kwargs"] = feature_extractor_kwargs
        pretrained_ckpt["recurrent_net_kwargs"] = recurrent_net_kwargs

        # Define optimizer
        optimizer = torch.optim.Adam(actor.parameters(), lr=args.pretrain_lr)

        print("\nStarting pretraining...")
        for epoch in range(1, args.pretrain_epochs):

            with tqdm(enumerate(data), total=len(data)) as tepoch:

                tepoch.set_description(f"Epoch {epoch}")

                for step, batch in tepoch:

                    # Separate batch data
                    ((scaffold_batch, scaffold_lengths), (decorator_batch, decorator_length)) = batch

                    # Prediction
                    encoded_seqs, rhs = actor.policy_net.memory_net._forward_encoder(
                        scaffold_batch.to(device), scaffold_lengths)
                    features, _, _ = actor.policy_net.memory_net._forward_decoder(
                        decorator_batch.to(device), decorator_length, encoded_seqs, rhs)
                    logp_action, entropy_dist, dist = actor.policy_net.dist.evaluate_pred(
                        features[:-1, :], decorator_batch.to(device)[1:, :])

                    # Optimization step
                    loss = - logp_action.squeeze(-1).sum(0).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    info_dict = {}
                    total_steps = step + len(data) * (epoch - 1)
                    if (total_steps % args.pretrain_lr_decrease_period) == 0 and total_steps != 0:

                        # Save model
                        pretrained_ckpt["network_weights"] = actor.state_dict()
                        torch.save(pretrained_ckpt, f"{args.log_dir}/pretrained_ckpt.prior")

                    tepoch.set_postfix(loss=loss.item())

                    # Wandb logging
                    info_dict.update({"pretrain_loss": loss.item()})
                    wandb.log(info_dict, step=total_steps)

    print("Finished!")
