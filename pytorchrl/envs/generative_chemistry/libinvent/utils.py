import os
import pandas as pd
from pytorchrl.agent.env import load_baselines_results


def get_num_unique_smiles(experiment_path):

    def unique_smiles():

        # Read monitor files
        data = load_baselines_results(os.path.join(experiment_path, "monitor_logs/train"))

        # Rank monitor files by reward
        data = data.sort_values("r", ascending=False)

        # Filter data
        data = data[data["molecule"] != "invalid_smile"]
        data = data[data["molecule"].duplicated() == False]
        data = data[data["r"] > 0.4]

        # Get number of unique smiles
        num_smiles = data.shape[0]

        return num_smiles

    return unique_smiles
