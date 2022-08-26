#!/usr/bin/env python3

import os
import re
import sys
import wget
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


# TODO: prior_trainingset is just a file with SMILES in it ("valid" SMILES)
# TODO: review SMILES filtered
# TODO: add wandb logging


def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def canonicalize_smiles_from_file(fname):
    """Reads a SMILES file and returns a list of RDKIT SMILES"""
    with open(fname, 'r') as f:
        smiles_list = []
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("{} lines processed.".format(i))
            smiles = line.split(" ")[0]
            mol = Chem.MolFromSmiles(smiles)
            if filter_mol(mol):
                smiles_list.append(Chem.MolToSmiles(mol))
        print("{} SMILES retrieved".format(len(smiles_list)))
        return smiles_list


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


def write_smiles_to_file(smiles_list, fname):
    """Write a list of SMILES to a file."""
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")


def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string


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
        return Variable(encoded)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_length = max([seq.size(0) for seq in arr])
        collated_arr = Variable(torch.zeros(len(arr), max_length))
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # Configuration file, keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile)

    # Wandb
    parser.add_argument(
        '--experiment_name', default=None, help='Name of the wandb experiment the agent belongs to')
    parser.add_argument(
        '--agent-name', default=None, help='Name of the wandb run')
    parser.add_argument(
        '--wandb-key', default=None, help='Init key from wandb account')

    # Environment specs
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action (default no skip)')
    parser.add_argument(
        '--frame-stack', type=int, default=0,
        help='Number of frame to stack in observation (default no stack)')

    # PPO specs
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps', type=float, default=1e-5,
        help='Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae', action='store_true', default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda', type=float, default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef', type=float, default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef', type=float, default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--use_clipped_value_loss', action='store_true', default=False,
        help='clip value loss update')
    parser.add_argument(
        '--num-steps', type=int, default=20000,
        help='number of forward steps in PPO (default: 20000)')
    parser.add_argument(
        '--ppo-epoch', type=int, default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch', type=int, default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param', type=float, default=0.2,
        help='ppo clip parameter (default: 0.2)')

    # Feature extractor model specs
    parser.add_argument(
        '--feature-extractor-net', default='MLP', help='Type of nn. Options include MLP, CNN, Fixup')
    parser.add_argument(
        '--restart-model', default=None,
        help='Restart training using the model given')
    parser.add_argument(
        '--recurrent-net', default=None, help='Recurrent neural networks to use')
    parser.add_argument(
        '--kl-coef', type=float, default=0.5,
        help='discount factor for rewards (default: 0.5)')

    # Scheme specs
    parser.add_argument(
        '--num-env-processes', type=int, default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-grad-workers', type=int, default=1,
        help='how many agent workers to use (default: 1)')
    parser.add_argument(
        '--com-grad-workers', default='synchronous',
        help='communication patters grad workers (default: synchronous)')
    parser.add_argument(
        '--num-col-workers', type=int, default=1,
        help='how many agent workers to use (default: 1)')
    parser.add_argument(
        '--com-col-workers', default='synchronous',
        help='communication patters col workers (default: synchronous)')
    parser.add_argument(
        '--cluster', action='store_true', default=False,
        help='script is running in a cluster')

    # General training specs
    parser.add_argument(
        '--num-env-steps', type=int, default=10e7,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--max-time', type=int, default=-1,
        help='stop script after this amount of time in seconds (default: no limit)')
    parser.add_argument(
        '--log-interval', type=int, default=1,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval', type=int, default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--log-dir', default='/tmp/obstacle_tower_ppo',
        help='directory to save agent logs (default: /tmp/obstacle_tower_ppo)')
    parser.add_argument(
        '--prior-trainingset-path', default=None, help='Path to dataset to train the prior')

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args


if __name__ == "__main__":

    args = get_args()
    cleanup_log_dir(args.log_dir)
    save_argparse(args, os.path.join(args.log_dir, "conf.yaml"), [])
    pretrained_ckpt = {}
    os.makedirs("data", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128

    original_data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.prior_trainingset_path)

    if not os.path.exists(original_data_path):
        raise ValueError(f"Missing training set: {original_data_path}")

    if not os.path.exists(f"{args.log_dir}/mols_filtered.smi"):
        print("Reading smiles...")
        smiles_list = canonicalize_smiles_from_file( original_data_path)
        print("Saving filtered training data...")
        write_smiles_to_file(smiles_list, f"{args.log_dir}/mols_filtered.smi")
    else:
        fname = f"{args.log_dir}/mols_filtered.smi"
        smiles_list = []
        with open(fname, 'r') as f:
            for line in f:
                smiles_list.append(line.split()[0])

    if not os.path.exists(f"{args.log_dir}/pretrained_ckpt.prior"):
        print("Constructing vocabulary...")
        tokenizer = SMILESTokenizer()
        vocabulary = create_vocabulary(smiles_list, tokenizer=tokenizer)
        pretrained_ckpt["tokenizer"] = tokenizer
        pretrained_ckpt["vocabulary"] = vocabulary
        torch.save(pretrained_ckpt, f"{args.log_dir}/pretrained_ckpt.prior")
    else:
        pretrained_ckpt_dict = torch.load(f"{args.log_dir}/pretrained_ckpt.prior")
        tokenizer = pretrained_ckpt_dict["tokenizer"]
        vocabulary = pretrained_ckpt_dict["vocabulary"]

    # Handle wandb init
    if args.wandb_key:
        mode = "online"
        wandb.login(key=str(args.wandb_key))
    else:
        mode = "disabled"

    with wandb.init(project=args.experiment_name, name=args.agent_name + "_pretrain", config=args, mode=mode):

        # Define Dataloader
        moldata = MolData(f"{args.log_dir}/mols_filtered.smi", vocabulary, tokenizer)
        data = DataLoader(moldata, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=MolData.collate_fn)

        # Define env
        _, action_space, obs_space = VecEnv.create_factory(
            env_fn=generative_chemistry_train_env_factory,
            env_kwargs={"scoring_function": lambda a: {"reward": 1.0}, "tokenizer": tokenizer, "vocabulary": vocabulary},
            vec_env_size=1)

        # Define model
        network_params = {'dropout': 0.0, 'layer_size': 512, 'num_layers': 3}
        actor = OnPolicyActor.create_factory(
            obs_space, action_space, prl.PPO,
            feature_extractor_network=get_feature_extractor(args.feature_extractor_net),
            feature_extractor_kwargs={"vocabulary_size": len(vocabulary)},
            recurrent_net=get_memory_network(args.recurrent_net),
            recurrent_net_kwargs={**network_params})(device)

        optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)

        for epoch in range(1, 6):
            # When training on a few million compounds, this model converges
            # in a few of epochs or even faster. If model sized is increased
            # its probably a good idea to check loss against an external set of
            # validation SMILES to make sure we dont overfit too much.
            for step, batch in tqdm(enumerate(data), total=len(data)):

                # Sample from DataLoader seqs = (batch_size, seq_length)
                seqs = batch.long()

                # Transpose seqs because memory net wants seqs = (seq_length, batch_size)
                seqs = torch.transpose(seqs, dim0=0, dim1=1)

                # Predict next token log likelihood
                # TODO. abstract this forward pass
                features = actor.policy_net.feature_extractor(seqs[0:-1])
                features, _ = actor.policy_net.memory_net._rnn(features)
                logp_action, entropy_dist, dist = actor.policy_net.dist.evaluate_pred(features, seqs[1:])

                # Optimization step
                loss = - logp_action.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Wandb logging
                wandb.log({"pretrain_loss": loss.item()}, step=learner.num_samples_collected)

    print("Finished!")
