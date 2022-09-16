#!/bin/bash
DIR=code_examples/train_genchem/sac
CUDA_VISIBLE_DEVICES="0" python $DIR/train_rnn_model.py -c $DIR/conf.yaml
