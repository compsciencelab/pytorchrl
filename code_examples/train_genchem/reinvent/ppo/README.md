# Generative chemistry in PyTorchRL

This code example uses the same approach as REINVENT (Blaschke et al., 2020) to guide a RL Proximal Policy Optimization (PPO) agent in the process of drug discovery.

## 1. Requirements

To be able to use the same scoring functions as REINVENT, install the conda environment. To use a customized scoring function that is not necessary.

    conda env create -f environment.yml
    conda activate reinvent

pytorchrl also has to be pip installed

    pip install pytorchrl

## 2. Experiment configuration

Training parameters can be adjusted in the `code_examples/train_genchem/reinvent/ppo/conf.yaml` file. Default parameters are reasonable values for the PPO algorithm. To get a description of each parameter run

    python code_examples/train_genchem/reinvent/ppo/train_rnn_model.py --help

## 3. Pre-training (not necessary)

To pretrain an agent with the current `conf.yaml` configuration run

    ./code_examples/train_genchem/reinvent/ppo/pretrain_rnn_model.sh

In particular, the `conf.yaml` field `pretrainingset_path` has to provide a valid path to a file with SMILES for the agent to be pre-trained on. Additionally,  `conf.yaml` fields `pretrain_element_list`,  `pretrain_min_heavy_atoms` and `pretrain_max_heavy_atoms` allow filtering out molecules from the training set.

## 4. Training

To train an agent with the current `conf.yaml` configuration run

    ./code_examples/train_genchem/reinvent/ppo/train_rnn_model.sh

If the agent was not pre-trained, a default pre-trained model is used.

## 5. Log in wandb during training

If you have a `wandb` account, you can visualise you training progress at https://wandb.ai/ by adding you account key to the configuration file in `line 35`.

## 6. Analize results

After training, results (including an image of the top generated molecules) can be analized by running

    ./code_examples/train_genchem/reinvent/ppo/analize_results.sh

## 7. Use a custom scoring function

By default, the training script will use a default scoring function, which can be found in 

    code_examples/train_genchem/reinvent/default_scoring_function.py

### 7.1 Define an alternative scoring function

Any custom scoring function can be used as long as it fulfills the following requirements:
    
- The method accepts a SMILE string as input and outputs a dict.
- The output dict contains at least a 1 key that represents the reward obtained (both keywords,"reward" and "score", are accepted).
- Optionally, the output dict can include more information about the molecules to be logged and tracked during training.

As a reference, an instance output for the default_scoring_function looks like that:

    output = {
        "reward": 0.85,
        "valid_smile": True,
        "QED_score": 0.64,
        "regression_model": 0.34,
        "matching_substructure": 0.53,
        "raw_regression_model": 0.33,
    }

Additionally, for a custom scoring function dummy example look at `code_examples/train_genchem/reinvent/dummy_custom_scoring_function.py`

### 7.2 Code Adjustments

To use a custom scoring function two minor code modifications are required.

First, replace `line 25` in `code_examples/train_genchem/reinvent/ppo/train_rnn_model.py` by importing the custom scoring_function. For example from 

    from default_scoring_function import scoring_function

to

    from myscript import my_scoring_function as scoring_function

Second, remove all the optional keywords from the default scoring_function output in `line 55` and replace them by the optional keywords of the custom scoring function. For example, if your scoring function has an output as follows

    output = {
        "reward": 0.85,
        "extra_value1": 0.4,
        "extra_value2": 0.58,
    }

modify the script in `line 55` from

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

Now the training script should work just like with the default scoring function!