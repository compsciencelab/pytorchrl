#!/bin/bash
DIR=code_examples/train_pybullet/ddpg
CUDA_VISIBLE_DEVICES="0" python $DIR/train.py -c $DIR/conf.yaml
