#!/usr/bin/env python3

import os
import time

from rdkit import Chem
from rdkit.Chem import Draw

from pytorchrl.agent.env import load_baselines_results
from code_examples.train_genchem.ppo.train import get_args


def analize_results(num_top_molecules=10):

    args = get_args()

    # Read monitor files
    monitor_files = load_baselines_results(os.path.join(args.log_dir, "monitor_logs/train"))

    # Rank monitor files by reward
    monitor_files = monitor_files.sort_values("r", ascending=False)

    # List top X molecules with highest score
    print(monitor_files[["r", "molecule"]].head(n=num_top_molecules))

    # Generate and save 2D smiles image
    smile_list = monitor_files['molecule'][:num_top_molecules].to_list()
    mols = [Chem.MolFromSmiles(smiles) for smiles in smile_list]
    image = Draw.MolsToGridImage(mols)
    save_name = os.path.join(args.log_dir, "2d_smiles.png")
    image.save(save_name)
    print(f"Saved 2d smile image as {save_name}")


if __name__ == "__main__":

    analize_results()
