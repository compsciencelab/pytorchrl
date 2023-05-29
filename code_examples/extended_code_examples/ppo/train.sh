#!/bin/bash
DIR=code_examples/extended_code_examples/ppo
CUDA_VISIBLE_DEVICES="0" python $DIR/train.py -c $DIR/conf.yaml
