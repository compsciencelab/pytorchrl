#!/bin/bash
DIR=code_examples/train_obstacle_tower/ppod
CUDA_VISIBLE_DEVICES="0" python $DIR/enjoy.py -c $DIR/conf.yaml
