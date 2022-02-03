#!/bin/bash
DIR=code_examples/train_mujoco/mpc
CUDA_VISIBLE_DEVICES="0" python $DIR/enjoy.py -c $DIR/conf.yaml
