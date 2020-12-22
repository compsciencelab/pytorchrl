#!/bin/bash

LOG_DIR=/tmp/pybullet_ppo
INTER=10000
DEVICES="0,1,2,3"
MAX_TIME=86400
STEPS=100000000000
ENV_ID=HalfCheetahBulletEnv-v0

###############################################################################

EPS=2e-4
LR=0.00025
MODEL=MLP
NUM_PROC=1
NUM_STEPS=2048
NUM_MINI_BATCH=32
CLIP_PARAM=0.2
GAMMA=0.99
PPO_EPOCH=10
GAE_LAMBDA=0.95
VALUE_LOSS_COEF=0.5
ENTROPY_COEF=0.0
FRAME_SKIP=0
FRAME_STACK=1
NUM_GRAD_WORKERS=1
COM_GRAD_WORKERS="synchronous"
NUM_COL_WORKERS=1
COM_COL_WORKERS="synchronous"

###############################################################################

CUDA_VISIBLE_DEVICES=$DEVICES python epython_scripts/train_pybullet/ppo/train.py  \
--lr $LR --clip-param $CLIP_PARAM --num-steps $NUM_STEPS --num-mini-batch $NUM_MINI_BATCH \
--num-env-steps $STEPS --log-dir $LOG_DIR --nn $MODEL --gamma $GAMMA --save-interval $INTER \
--ppo-epoch $PPO_EPOCH --gae-lambda $GAE_LAMBDA --use-gae --num-env-processes $NUM_PROC \
--value-loss-coef $VALUE_LOSS_COEF --entropy-coef $ENTROPY_COEF --eps $EPS --max-time $MAX_TIME \
--use_clipped_value_loss --frame-skip $FRAME_SKIP --frame-stack $FRAME_STACK --env-id $ENV_ID \
--num-grad-workers $NUM_GRAD_WORKERS --num-col-workers $NUM_COL_WORKERS \
--com-grad-workers $COM_GRAD_WORKERS --com-col-workers $COM_COL_WORKERS