#!/usr/bin/env python3

import os
import time

from rdkit import Chem
from rdkit.Chem import Draw

from pytorchrl.agent.env import load_baselines_results


def analize_results(experimentpath, num_top_molecules=10):
    """ Tries to run again failed agents in a given experiment. """

    # Read monitor files
    monitor_files = load_baselines_results(os.path.join(experimentpath, "monitor_logs/train"))

    # Rank monitor files by reward
    monitor_files = monitor_files.sort_values("r", ascending=False)

    # List top X molecules with highest score
    print(monitor_files[["r", "molecules"]].head(n=num_top_molecules))

    # Generate and save 2D smiles image
    smile_list = monitor_files['molecules'][:num_top_molecules].to_list()
    mols = [Chem.MolFromSmiles(smiles) for smiles in smile_list]
    image = Draw.MolsToGridImage(mols)
    save_name = os.path.join(experimentpath, "2d_smiles.png")
    image.save(save_name)
    print(f"Saved images as {save_name}")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--experimentpath", default="/tmp/genchem_ppo/",
        help="List of directories containing experiments to be plotted")
    args = parser.parse_args()

    analize_results(experimentpath=args.experimentpath)
