#!/bin/bash
DIR=code_examples/train_obstacle_tower/ppo
CUDA_VISIBLE_DEVICES="0" python $DIR/enjoy.py -c $DIR/train.yaml
