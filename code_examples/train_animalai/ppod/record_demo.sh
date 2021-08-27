#!/bin/bash
# SAVE_DIR=/Users/abou/PycharmProjects/pytorchrl/code_examples/train_animalai/ppod
SAVE_DIR=/tmp/demos/
FRAME_SKIP=2
FRAME_STACK=4

python code_examples/train_animalai/ppod/record_demo.py \
--frame-skip $FRAME_SKIP --frame-stack $FRAME_STACK --save-dir $SAVE_DIR