#!/bin/bash
DIR=code_examples/train_genchem/ppo
CUDA_VISIBLE_DEVICES="0" python $DIR/pretrain_2.py -c $DIR/conf.yaml
