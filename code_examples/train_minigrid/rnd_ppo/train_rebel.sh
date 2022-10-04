#!/bin/bash
DIR=code_examples/train_minigrid/rnd_ppo
CUDA_VISIBLE_DEVICES="0" python $DIR/train_rebel.py -c $DIR/conf.yaml
