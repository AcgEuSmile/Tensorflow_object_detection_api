#!/bin/bash

mkdir -p eval_logs
CUDA_VISIBLE_DEVICES="-1" python ./eval.py \
  --logtostderr \
  --pipeline_config_path=./faster_rcnn_nas_coco_setting/faster_rcnn_nas_coco.config\
  --checkpoint_dir=train_logs \
  --eval_dir=eval_logs &

#python /workspace/yo/tensorflow_obj_detect_api/google_obj_detection/research/object_detection/legacy/eval.py \
#  --logtostderr \
#  --pipeline_config_path=./faster_rcnn_nas_coco_setting/faster_rcnn_nas_coco.config\
#  --checkpoint_dir=train_logs \
#  --eval_dir=eval_logs &
