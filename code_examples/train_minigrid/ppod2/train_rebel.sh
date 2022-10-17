#!/bin/bash
DIR=code_examples/train_minigrid/ppod2
CUDA_VISIBLE_DEVICES="0" python $DIR/train_rebel.py -c $DIR/conf.yaml
