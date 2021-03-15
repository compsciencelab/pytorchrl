#!/bin/bash
DIR=code_examples/train_pybullet/td3
CUDA_VISIBLE_DEVICES="0" python $DIR/enjoy.py -c $DIR/conf.yaml
