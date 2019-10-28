#!/bin/bash

mkdir -p logs/
now=$(date +"%Y_%m_%d")
CUDA_VISIBLE_DEVICES=0,1 python /workspace/yo/tensorflow_obj_detect_api/google_obj_detection/research/object_detection/legacy/train.py \
  --logtostderr \
  --num_clones=2 \
  --ps_tasks=1 \
  --pipeline_config_path=./faster_rcnn_nas_coco_setting/faster_rcnn_nas_coco.config \
  --train_dir=train_logs 2>&1 | tee logs/train_$now.txt &
