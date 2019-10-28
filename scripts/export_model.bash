#!/bin/bash

mkdir -p output

CUDA_VISIBLE_DEVICES="0,1" python /workspace/yo/tensorflow_obj_detect_api/google_obj_detection/research/object_detection/export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path faster_rcnn_nas_coco_setting/faster_rcnn_nas_coco.config \
  --trained_checkpoint_prefix train_logs/model.ckpt-141015\
  --output_directory output/
