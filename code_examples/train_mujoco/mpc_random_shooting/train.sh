#!/bin/bash
DIR=code_examples/train_mujoco/mpc_random_shooting
CUDA_VISIBLE_DEVICES="0" python $DIR/train.py -c $DIR/conf.yaml
