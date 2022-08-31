#!/usr/bin/env python3

"""Code adapted from https://github.com/MarcusOlivecrona/REINVENT"""

import os
import re
import sys
import time
import wandb
import torch
import argparse
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from torch.utils.data import Dataset
from pytorchrl.agent.env import VecEnv
from torch.utils.data import DataLoader

import pytorchrl as prl
from pytorchrl.utils import LoadFromFile, save_argparse, cleanup_log_dir
from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor, get_memory_network
from pytorchrl.envs.generative_chemistry.vocabulary import SMILESTokenizer, create_vocabulary
from pytorchrl.envs.generative_chemistry.generative_chemistry_env_factory import generative_chemistry_train_env_factory
from code_examples.train_genchem.ppo.train import get_args


def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)


def is_valid_smile(smile):
    """Returns true is smile is syntactically valid."""
    mol = Chem.MolFromSmiles(smile)
    return mol is not None


def filter_mol(mol, max_heavy_atoms=50, min_heavy_atoms=10, element_list=[6, 7, 8, 9, 16, 17, 35]):
    """
    Filters molecules on number of heavy atoms and atom types.

    element_list: to filter out smiles that contain atoms of other elements.
    """
    if mol is not None:
        num_heavy = min_heavy_atoms < mol.GetNumHeavyAtoms() < max_heavy_atoms
        elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        if num_heavy and elements:
            return True
        else:
            return False


def read_and_filter_data(fname, max_heavy_atoms, min_heavy_atoms, element_list):
    """Reads a SMILES file and returns a list of RDKIT SMILES"""
    with open(fname, 'r') as f:
        smiles_list = []
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("{} lines processed.".format(i))
            smiles = line.split(" ")[0]
            mol = Chem.MolFromSmiles(smiles)
            if filter_mol(mol, max_heavy_atoms, min_heavy_atoms, element_list):
                smiles_list.append(Chem.MolToSmiles(mol))
        print("{} SMILES retrieved".format(len(smiles_list)))
        return smiles_list


def write_smiles_to_file(smiles_list, fname):
    """Write a list of SMILES to a file."""
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")


class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

    Args:
            fname : path to a file containing \n separated SMILES.
            voc   : a Vocabulary instance
    Returns:
            A custom PyTorch dataset for training the Prior.
    """
    def __init__(self, fname, voc, tokenizer):
        self.voc = voc
        self.tokenizer = tokenizer
        self.smiles = []
        with open(fname, 'r') as f:
            for line in f:
                self.smiles.append(line.split()[0])

    def __getitem__(self, i):
        mol = self.smiles[i]
        tokenized = self.tokenizer.tokenize(mol)
        encoded = self.voc.encode(tokenized)
        return torch.from_numpy(encoded)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_length = max([seq.size(0) for seq in arr])
        collated_arr = torch.zeros(len(arr), max_length)
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr


if __name__ == "__main__":

    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)
    save_argparse(args, os.path.join(args.log_dir, "conf.yaml"), [])
    pretrained_ckpt = {}
    os.makedirs("data", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(f"{args.log_dir}/mols_filtered.smi"):
        if not os.path.exists(args.pretrainingset_path):
            raise ValueError(f"Missing training set: {args.pretrainingset_path}")
        print("\nReading smiles...")
        smiles_list = read_and_filter_data(
            args.pretrainingset_path,
            args.pretrain_max_heavy_atoms,
            args.pretrain_min_heavy_atoms,
            args.pretrain_element_list,
        )
        print("\nSaving filtered training data...")
        write_smiles_to_file(smiles_list, f"{args.log_dir}/mols_filtered.smi")
    else:
        fname = f"{args.log_dir}/mols_filtered.smi"
        smiles_list = []
        with open(fname, 'r') as f:
            for line in f:
                smiles_list.append(line.split()[0])

    if not os.path.exists(f"{args.log_dir}/pretrained_ckpt.prior"):
        print("\nConstructing vocabulary...")
        tokenizer = SMILESTokenizer()
        vocabulary = create_vocabulary(smiles_list, tokenizer=tokenizer)
        pretrained_ckpt["tokenizer"] = tokenizer
        pretrained_ckpt["vocabulary"] = vocabulary
        pretrained_ckpt["max_sequence_length"] = args.pretrain_max_smile_length
        torch.save(pretrained_ckpt, f"{args.log_dir}/pretrained_ckpt.prior")
    else:
        pretrained_ckpt_dict = torch.load(f"{args.log_dir}/pretrained_ckpt.prior")
        tokenizer = pretrained_ckpt_dict["tokenizer"]
        vocabulary = pretrained_ckpt_dict["vocabulary"]
        pretrained_ckpt["max_sequence_length"] = args.pretrain_max_smile_length

    # Handle wandb init
    if args.wandb_key:
        mode = "online"
        wandb.login(key=str(args.wandb_key))
    else:
        mode = "disabled"

    with wandb.init(project=args.experiment_name, name=args.agent_name + "_pretrain", config=args, mode=mode):

        # Define Dataloader
        moldata = MolData(f"{args.log_dir}/mols_filtered.smi", vocabulary, tokenizer)
        data = DataLoader(
            moldata, batch_size=args.pretrain_batch_size, shuffle=True, drop_last=True, collate_fn=MolData.collate_fn)

        # Define env
        test_env, action_space, obs_space = VecEnv.create_factory(
            env_fn=generative_chemistry_train_env_factory,
            env_kwargs={
                "scoring_function": lambda a: {"reward": 1.0},
                "tokenizer": tokenizer, "vocabulary": vocabulary,
                "smiles_max_length": args.pretrain_max_smile_length},
            vec_env_size=1)
        env = test_env(device)

        # Define model
        feature_extractor_kwargs = {"vocabulary_size": len(vocabulary)}
        recurrent_net_kwargs = {}
        actor = OnPolicyActor.create_factory(
            obs_space, action_space, prl.PPO,
            feature_extractor_network=get_feature_extractor(args.feature_extractor_net),
            feature_extractor_kwargs={**feature_extractor_kwargs},
            recurrent_net=get_memory_network(args.recurrent_net),
            recurrent_net_kwargs={**recurrent_net_kwargs})(device)
        pretrained_ckpt["feature_extractor_kwargs"] = feature_extractor_kwargs
        pretrained_ckpt["recurrent_net_kwargs"] = recurrent_net_kwargs

        # Define optimizer
        optimizer = torch.optim.Adam(actor.parameters(), lr=args.pretrain_lr)

        print("\nStarting pretraining...")
        for epoch in range(1, 10):
            # When training on a few million compounds, this model converges
            # in a few of epochs or even faster. If model sized is increased
            # its probably a good idea to check loss against an external set of
            # validation SMILES to make sure we dont overfit too much.
            for step, batch in tqdm(enumerate(data), total=len(data)):

                # Sample from DataLoader seqs = (batch_size, seq_length)
                seqs = batch.long().to(device)

                # Transpose seqs because memory net wants seqs = (seq_length, batch_size)
                seqs = torch.transpose(seqs, dim0=0, dim1=1)

                # Predict next token log likelihood. TODO: Ugly hack, abstract this forward pass
                features = actor.policy_net.feature_extractor(seqs[:-1, :])
                features, _ = actor.policy_net.memory_net._rnn(features)
                logp_action, entropy_dist, dist = actor.policy_net.dist.evaluate_pred(features, seqs[1:, :])

                # Optimization step
                loss = - logp_action.squeeze(-1).sum(0).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                info_dict = {}
                total_steps = step + len(data) * (epoch - 1)
                if (total_steps % args.pretrain_lr_decrease_period) == 0 and total_steps != 0:

                    # Decrease learning rate
                    decrease_learning_rate(optimizer, decrease_by=args.pretrain_lr_decrease_value)

                    # Generate a few molecules and check how many are valid
                    total_molecules = 100
                    valid_molecules = 0
                    list_molecules = []
                    list_tokens = []
                    list_entropy = []
                    for i in range(total_molecules):
                        obs, rhs, done = actor.actor_initial_states(env.reset())
                        tokens = []
                        while not done:
                            with torch.no_grad():
                                _, action, _, rhs, entropy_dist, dist = actor.get_action(
                                    obs, rhs, done, deterministic=False)
                            obs, _, done, _ = env.step(action)
                            tokens.append(vocabulary.decode([int(action)])[0])
                        molecule = tokenizer.untokenize(tokens)
                        if is_valid_smile(molecule):
                            valid_molecules += 1
                        list_molecules.append(molecule)
                        list_tokens.append(tokens)
                        list_entropy.append(entropy_dist.item())

                    # Check how many are repeated
                    ratio_repeated = len(set(list_molecules)) / len(list_molecules) if total_molecules > 0 else 0

                    # Add to info dict
                    info_dict.update({
                        "pretrain_avg_molecular_length": np.mean([len(s) for s in list_tokens]),
                        "pretrain_avg_entropy": np.mean(list_entropy),
                        "pretrain_valid_molecules": valid_molecules / total_molecules,
                        "pretrain_ratio_repeated": ratio_repeated
                    })

                    # Save model
                    pretrained_ckpt["network_weights"] = actor.state_dict()
                    torch.save(pretrained_ckpt, f"{args.log_dir}/pretrained_ckpt.prior")

                # Wandb logging
                info_dict.update({"pretrain_loss": loss.item()})
                wandb.log(info_dict, step=total_steps)

    print("Finished!")
