#!/bin/bash
DIR=code_examples/train_minigrid/rnd_ppo
CUDA_VISIBLE_DEVICES="0" python $DIR/enjoy_policy.py -c $DIR/conf.yaml
