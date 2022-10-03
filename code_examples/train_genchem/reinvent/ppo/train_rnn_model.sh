#!/bin/bash
<<<<<<< HEAD
DIR=code_examples/train_genchem/reinvent/ppo
=======
DIR=code_examples/train_genchem/ppo_reinvent
>>>>>>> 0dcf0ed (genchem code examples)
CUDA_VISIBLE_DEVICES="0" python $DIR/train_rnn_model.py -c $DIR/conf.yaml
