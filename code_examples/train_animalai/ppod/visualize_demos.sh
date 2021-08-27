#!/bin/bash
DIR=code_examples/train_animalai/ppod
# SAVE_DIR=/Users/abou/PycharmProjects/pytorchrl/code_examples/train_animalai/ppod
SAVE_DIR=/tmp/demos/
FRAME_SKIP=2
FRAME_STACK=4

python code_examples/train_animalai/ppod/visualize_demos.py -c $DIR/conf.yaml