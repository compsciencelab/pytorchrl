#!/bin/bash
DIR=code_examples/train_minigrid/ppo
CUDA_VISIBLE_DEVICES="0" python $DIR/play.py -c $DIR/conf.yaml
