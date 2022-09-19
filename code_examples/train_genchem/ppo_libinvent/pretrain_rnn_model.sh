#!/bin/bash
DIR=code_examples/train_genchem/ppo_libinvent
CUDA_VISIBLE_DEVICES="0" python $DIR/pretrain_rnn_model.py -c $DIR/conf.yaml
