#!/bin/bash
DIR=code_examples/train_atari/rnd_ppod
CUDA_VISIBLE_DEVICES="0" python $DIR/enjoy.py -c $DIR/conf.yaml
