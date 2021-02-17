#!/bin/bash

LOG_DIR=/tmp/atari_sac
INTER=50000
DEVICES="0,1,2,3"
MAX_TIME=86400
STEPS=10000000000000
ENV_ID=MsPacmanNoFrameskip-v4

###############################################################################

LR=0.0003
MODEL=CNN
NUM_PROC=1
GAMMA=0.99
POLYAK=0.995
MINI_BATCH_SIZE=64
UPDATE_EVERY=4
NUM_UPDATES=1
START_STEPS=20000
BUFFER_SIZE=300000
FRAME_SKIP=4
FRAME_STACK=4
TARGET_UPDATE_INTERVAL=8000
NUM_GRAD_WORKERS=1
COM_GRAD_WORKERS="synchronous"
NUM_COL_WORKERS=1
COM_COL_WORKERS="synchronous"

###############################################################################

CUDA_VISIBLE_DEVICES=$DEVICES python python_scripts/train_atari/sac/train.py  \
--num-env-steps $STEPS --log-dir $LOG_DIR --gamma $GAMMA --polyak $POLYAK \
--save-interval $INTER --max-time $MAX_TIME --num-env-processes $NUM_PROC \
--mini-batch-size $MINI_BATCH_SIZE --update-every $UPDATE_EVERY --num-updates $NUM_UPDATES \
--start-steps $START_STEPS --buffer-size $BUFFER_SIZE --lr $LR --nn $MODEL \
--frame-skip $FRAME_SKIP --frame-stack $FRAME_STACK --env-id $ENV_ID \
--target-update-interval $TARGET_UPDATE_INTERVAL \
--num-grad-workers $NUM_GRAD_WORKERS --num-col-workers $NUM_COL_WORKERS \
--com-grad-workers $COM_GRAD_WORKERS --com-col-workers $COM_COL_WORKERS
