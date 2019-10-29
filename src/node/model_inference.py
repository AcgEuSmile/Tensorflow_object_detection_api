import sys
import os
import json
import time
import tensorflow as tf
import numpy as np
import glob
import argparse
from PIL import Image
from object_detection.utils import label_map_util
# import self package library
sys.path.append('/workspace/yo/google_od_api/src/lib/')
from utility import load_config
        
parser = argparse.ArgumentParser(description='model inference')
parser.add_argument("--config_path", type = str,
                    default="cfg/model_inference.json", help="config path")
args = parser.parse_args()
cfg = load_config.readCfg(args.config_path)

def get_results(boxes, classes, scores, category_index, im_width, im_height,
    min_score_thresh=.2):
    bboxes = list()
    for i, box in enumerate(boxes):
        if scores[i] > min_score_thresh:
            ymin, xmin, ymax, xmax = box
            bbox = {
                'bbox': {
                    'xmax': xmax * im_width,
                    'xmin': xmin * im_width,
                    'ymax': ymax * im_height,
                    'ymin': ymin * im_height
                },
                'category': category_index[classes[i]]['name'],
                'score': float(scores[i])
            }
            bboxes.append(bbox)
    return bboxes

def main():
    label_map = label_map_util.load_labelmap(cfg["PATH_TO_LABELS"])
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=cfg["NUM_CLASSES"], use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(cfg["PATH_TO_CKPT"], 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    test_annos = dict()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            for index, image_path in enumerate(glob.glob(cfg["DIR_IMAGE"]+'/*.jpg')):
                image_id = image_path.rstrip().split('/')[-1]
                image = Image.open(image_path)
                image_np = np.array(image).astype(np.uint8)
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
                test_annos[image_id] = {'objects': get_results(
                    np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                    category_index, im_width, im_height)}
                if index > 10:
                    break

    test_annos = {'imgs': test_annos}
    fd = open(cfg["PATH_OUTPUT"], 'w')
    json.dump(test_annos, fd)
    fd.close()
    print("success")

if __name__ == "__main__":
    main()