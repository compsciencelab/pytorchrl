#!/bin/bash
DIR=python_scripts/train_pybullet/sac
CUDA_VISIBLE_DEVICES="0" python $DIR/train.py -c $DIR/train.yaml
