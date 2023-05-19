#!/bin/bash
DIR=extended_code_examples/train_atari/ppo
CUDA_VISIBLE_DEVICES="0" python $DIR/train.py -c $DIR/conf.yaml
