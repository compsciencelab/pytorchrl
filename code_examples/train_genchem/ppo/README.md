# Generative chemistry in PyTorchRL

This code example uses the same approach as REINVENT (Blaschke et al., 2020) to guide a RL Proximal Policy Optimization (PPO) agent in the process of drug discovery.

## 1. Requirements

To be able to use the same scoring functions as REINVENT, install conda environment from REINVENT github repository. To use a customized scoring function that is not necessary.

    https://github.com/MolecularAI/Reinvent

pytorchrl also has to be pip installed

    pip install pytorchrl

## 2. Experiment configuration

Training parameters can be adjusted in the code_examples/train_genchem/ppo/conf.yaml file. Defaults parameters are reasonable values for the PPO algorithm. To get a description of each one of them run

    python code_examples/train_genchem/ppo/train.py --help

## 3. Training

To train an agent with the current ´conf.yaml´ configuration run

    ./code_examples/train_genchem/ppo/train.sh

## 4. Log in wandb during training

If you have a wandb account, you can visualise you training progress in https://wandb.ai/ by adding you account key to the configuration file.

## 5. Analize results

After training, results (including an image of the top generated molecules) can be analized by running

    ./code_examples/train_genchem/ppo/analize_results.sh

## 6. Use a custom scoring function

By default, the training script will use a default scoring function, which can be found in 

    pytorchrl/envs/generative_chemistry/default_scoring_function.py

### 6.1 Define an alternative scoring function

A custom scoring function can be used as long as it fulfills the following requirements:
    
- The method accepts a SMILE string as input and outputs a dict.
- The output dict contains at least a 1 key that represents the reward obtained (both keywords,"reward" and "score", are accepted)
- optionally, the output dict can include more information about the molecules to be logged and tracked. during training

An output example for the scoring_function output is the following:

    output = {
        "reward": 0.85,
        "valid_smile": True,
        "QED_score": 0.64,
        "regression_model": 0.34,
        "matching_substructure": 0.53,
        "raw_regression_model": 0.33,
    }

### 6.2 Code Adjustments

To use a custom scoring function two minor code modifications are required

First, replace line 25 in code_examples/train_genchem/ppo/train.py by importing the custom scoring_function, for example

from 

    from pytorchrl.envs.generative_chemistry.default_scoring_function import scoring_function

to

    from myscript import my_scoring_function as scoring_function

Second, remove all the optional keywords from the default scoring_function output in line 55 and replace them by the optional keywords of the custom scoring function

For example if your scoring function has an output as follows

    output = {
        "reward": 0.85,
        "extra_value1": 0.4,
        "extra_value2": 0.58,
    }

modify the script in line 55 from

    info_keywords += (
        "regression_model",
        "matching_substructure",
        "custom_alerts",
        "QED_score",
        "raw_regression_model",
        "valid_smile"
    )

to

    info_keywords += (
        "extra_value1",
        "extra_value2",
    )
