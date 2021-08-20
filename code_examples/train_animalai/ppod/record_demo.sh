#!/bin/bash
SAVE_DIR=/Users/abou/PycharmProjects/pytorchrl/code_examples/train_sparse_reacher/ppod
FRAME_SKIP=2
FRAME_STACK=4

python code_examples/train_sparse_reacher/ppod/record_demo.py \
--frame-skip $FRAME_SKIP --frame-stack $FRAME_STACK --save-dir $SAVE_DIR