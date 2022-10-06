#!/usr/bin/env python3

import os
import time
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from pytorchrl.agent.env import load_baselines_results
from code_examples.train_genchem.reinvent.ppo.train_rnn_model import get_args


def analize_results(num_top_molecules=30):

    args = get_args()

    # Read monitor files
    monitor_files = load_baselines_results(os.path.join(args.log_dir, "monitor_logs/train"))

    # Rank monitor files by reward
    monitor_files = monitor_files.sort_values("r", ascending=False)
    total_proposed_molecules = monitor_files.shape[0]

    # List top X molecules with highest score
    pd.options.display.max_colwidth = 300
    monitor_files = monitor_files[monitor_files["molecule"].duplicated() == False]
    total_unique_molecules = monitor_files.shape[0]
    unique_percentage = (total_unique_molecules / total_proposed_molecules) * 100

    print()
    print(f"=" * 50)
    print(f"\nA total of {total_proposed_molecules} molecules was proposed by the agent, "
          f"of them {total_unique_molecules} were unique ({unique_percentage:.2f}%).")
    print(f"Here's a list of the top {num_top_molecules} molecules with the obtained reward (r):\n")
    print(monitor_files[["r", "molecule"]].head(n=num_top_molecules))

    # Generate and save 2D smiles image
    smile_list = monitor_files['molecule'][:num_top_molecules].to_list()
    mols = [Chem.MolFromSmiles(smiles) for smiles in smile_list]
    image = Draw.MolsToGridImage(mols)
    save_name = os.path.join(args.log_dir, "2d_smiles.png")
    image.save(save_name)
    print(f"\nA 2d smile image of the top molecules was generated and saved as {save_name}\n")
    print(f"=" * 50)


if __name__ == "__main__":

    analize_results()
