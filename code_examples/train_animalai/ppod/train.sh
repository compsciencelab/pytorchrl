#!/bin/bash
DIR=code_examples/tain_animalai/ppod
CUDA_VISIBLE_DEVICES="0" python $DIR/train.py -c $DIR/conf.yaml
