import sys
import cv2
import os
import json
import time
import tensorflow as tf
import numpy as np
import argparse
from PIL import Image
from object_detection.utils import label_map_util
# import self package library
sys.path.append('/workspace/yo/google_od_api/src/lib/')
from utility import load_config

parser = argparse.ArgumentParser(description='inference videos')
parser.add_argument("--config_path", type = str,
                    default="cfg/inference_vids.json", help="config path")
args = parser.parse_args()
CONFIG = load_config.readCfg(args.config_path)

ColorTable = dict({'RED': (0, 0, 255),
                  'ORANGE': (0, 165, 255),
                  'YELLOW': (0, 255, 255),
                  'GREEN': (0, 255, 0),
                  'BLUE': (255, 127, 0),
                  'INDIGO': (255, 0, 0),
                  'PURPLE': (255, 0, 139),
                  'WHITE': (255, 255, 255),
                  'BLACK': (0, 0, 0)}
)
ClassColor = dict(
        default = {'bike': ColorTable['RED'],
                   'bus': ColorTable['ORANGE'],
                   'car': ColorTable['YELLOW'],
                   'motor': ColorTable['GREEN'],
                   'person': ColorTable['BLUE'],
                   'rider': ColorTable['INDIGO'],
                   'truck': ColorTable['PURPLE'],
                 }
)

def get_results(boxes, classes, scores, category_index, im_width, im_height,
    min_score_thresh=.2):
    bboxes = list()
    for i, box in enumerate(boxes):
        if scores[i] > min_score_thresh:
            ymin, xmin, ymax, xmax = box
            bbox = {
                'bbox': {
                    'xmax': int(xmax * im_width),
                    'xmin': int(xmin * im_width),
                    'ymax': int(ymax * im_height),
                    'ymin': int(ymin * im_height)
                },
                'category': category_index[classes[i]]['name'],
                'score': float(scores[i])
            }
            bboxes.append(bbox)
    return bboxes

vcap = cv2.VideoCapture(CONFIG["VIDEO_PATH"])
frame_width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = int(vcap.get(cv2.CAP_PROP_FPS))
# fourcc = int(vcap.get(cv2.CAP_PROP_FOURCC))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(CONFIG["OUTPUT_PATH"], fourcc, frame_fps, (frame_width, frame_height))

label_map = label_map_util.load_labelmap(CONFIG["PATH_TO_LABELS"])
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=CONFIG["NUM_CLASSES"], use_display_name=True)
category_index = label_map_util.create_category_index(categories)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(CONFIG["PATH_TO_CKPT"], 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
        while(vcap.isOpened()):
          ret, frame = vcap.read()
          if ret == True:
            image_np = np.array(frame).astype(np.uint8)
            im_height, im_width, _ = image_np.shape
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            bboxes = get_results(np.squeeze(boxes), np.squeeze(classes).astype(np.int32),
              np.squeeze(scores), category_index, im_width, im_height)
            for bbox in bboxes:
              if(bbox['score']>CONFIG["THRESHOLD"]):
                cv2.rectangle(frame, (bbox['bbox']['xmin'], bbox['bbox']['ymax']),
                              (bbox['bbox']['xmax'], bbox['bbox']['ymin']),
                              ClassColor[CONFIG["CLASS_COLOR"]][bbox['category']], 2)
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                font_scale = 1
                thickness = 1
                margin = 5
                size = cv2.getTextSize(bbox['category'], font, font_scale, thickness)
                text_width = size[0][0]
                text_height = size[0][1]
                cv2.rectangle(frame, (bbox['bbox']['xmin'], bbox['bbox']['ymax']),
                              (bbox['bbox']['xmin']+text_width, bbox['bbox']['ymax']-text_height),
                              (0, 0, 0), thickness = -1)
                
                cv2.putText(frame, bbox['category'], (bbox['bbox']['xmin'], bbox['bbox']['ymax']),
                            font, 1, ClassColor[CONFIG["CLASS_COLOR"]][bbox['category']], 1, cv2.LINE_AA)
            out.write(frame)
          else:
            break 
                        
print("success")
vcap.release()
out.release()
cv2.destroyAllWindows()