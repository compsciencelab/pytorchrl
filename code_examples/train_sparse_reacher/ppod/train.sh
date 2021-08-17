#!/bin/bash
DIR=code_examples/train_sparse_reacher/ppod
CUDA_VISIBLE_DEVICES="0" python $DIR/train.py -c $DIR/conf.yaml
