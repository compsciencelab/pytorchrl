#!/bin/bash
DIR=code_examples/train_genchem/ppo_reinvent
CUDA_VISIBLE_DEVICES="0" python $DIR/pretrain_transformer_model.py -c $DIR/conf.yaml