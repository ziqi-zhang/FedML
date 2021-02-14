#!/usr/bin/env bash

CLIENT_NUM=10
WORKER_NUM=10
SERVER_NUM=1
GPU_NUM_PER_SERVER=1
MODEL=resnet20
DISTRIBUTION=homo
EPOCH=50
BATCH_SIZE=128
LR=0.1
DATASET=cifar10
DATA_DIR=./../../data/cifar10
CLIENT_OPTIMIZER=sgd
CI=0
GPU=0

echo $BATCH_SIZE
echo $LR

CUDA_VISIBLE_DEVICES=$1 \
python ./main.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --epochs $EPOCH \
  --client_optimizer $CLIENT_OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --ci $CI \
  --gpu $GPU \
  --frequency_of_test_acc_report 1 \
  --run_tag baseline \















