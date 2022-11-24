#!/bin/bash
DIR=code_examples/train_genchem/libinvent/ppo
CUDA_VISIBLE_DEVICES="0" python $DIR/train_rnn_model_batched.py -c $DIR/conf.yaml
