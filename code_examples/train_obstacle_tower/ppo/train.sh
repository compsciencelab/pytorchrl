#!/bin/bash
DIR=code_examples/train_obstacle_tower/ppo
CUDA_VISIBLE_DEVICES="0" python $DIR/train.py -c $DIR/train.yaml
