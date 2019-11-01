from __future__ import absolute_import

import os
import sys
import io
import pandas as pd
import tensorflow as tf
import json
import argparse

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# import self package library start
dirname = (os.path.dirname(os.path.realpath(__file__))).split("/")
dirname[-1] = "lib"
lib_path = "/".join((x for x in dirname))
sys.path.append(lib_path)
from utility import load_config
# import self package library end

parser = argparse.ArgumentParser(description='generator tf_record file')
parser.add_argument("--config_path", type = str,
                    default="cfg/generate_tfrecord_config.json", help="config path")
args = parser.parse_args()
cfg = load_config.readCfg(args.config_path)

def categoryText2Int(label):
    if label == "bike":
        return 1
    elif label == "bus":
        return 2
    elif label == "car":
        return 3
    elif label == "motor":
        return 4
    elif label == "person":
        return 5
    elif label == "rider":
        return 6
    elif label == "truck":
        return 7
    else:
        raise("Category error!!")

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def createTfRecord(group, img_path):
    with tf.gfile.GFile(os.path.join(img_path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encode_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encode_jpg_io)
    width, height = image.size
    filename = (group.filename).encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    for index, row in group.object.iterrows():
        xmins.append(row['xmin']/width)
        xmaxs.append(row['xmax']/width)
        ymins.append(row['ymin']/height)
        ymaxs.append(row['ymax']/height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(categoryText2Int(row['class']))
    
    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
    return tf_record

def main():
    writer = tf.python_io.TFRecordWriter(cfg["out_path"])
    csv_fd = pd.read_csv(cfg["csv_path"])
    grouped = split(csv_fd, 'filename')
    for group in grouped:
        tf_record = createTfRecord(group, cfg["img_path"])
        writer.write(tf_record.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords: {}'.format(cfg["out_path"]))

if __name__ == "__main__":
    main()
