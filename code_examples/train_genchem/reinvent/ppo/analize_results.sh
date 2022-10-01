#!/bin/bash
DIR=code_examples/train_genchem/reinvent/ppo
CUDA_VISIBLE_DEVICES="0" python $DIR/analize_results.py -c $DIR/conf.yaml
