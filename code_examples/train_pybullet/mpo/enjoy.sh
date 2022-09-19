#!/bin/bash
DIR=code_examples/train_pybullet/mpo
CUDA_VISIBLE_DEVICES="0" python $DIR/enjoy.py -c $DIR/conf.yaml
