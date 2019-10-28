#!/bin/bash
GOOGLE_OBJ_DETECTION_API_PATH=/workspace/yo/tensorflow_obj_detect_api/google_obj_detection/research:/workspace/yo/tensorflow_obj_detect_api/google_obj_detection/research/slim

echo "# Setting google object detection api path" >> ~/.bashrc
echo "export PYTHONPATH=${PYTHONPATH:-${GOOGLE_OBJ_DETECTION_API_PATH}}" >> ~/.bashrc
