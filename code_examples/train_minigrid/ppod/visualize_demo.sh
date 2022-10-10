#!/bin/bash
DIR=code_examples/train_minigrid/ppod
CUDA_VISIBLE_DEVICES="0" python $DIR/visualize_demo.py -c $DIR/conf.yaml
