# Generative chemistry in PyTorchRL

This code example uses the same approach as REINVENT (Blaschke et al., 2020) to guide a RL Proximal Policy Optimization (PPO) agent in the process of drug discovery.

## 1. Requirements

To be able to use the same scoring functions as REINVENT, install the conda environment. To use a customized scoring function that is not necessary.

    conda env create -f environment.yml
    conda activate reinvent.v3.2

pytorchrl also has to be pip installed

    pip install pytorchrl
