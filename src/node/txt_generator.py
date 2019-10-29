#!/usr/bin/env python
import json
import os
import glob
import sys
import argparse
from xml.etree.ElementTree import ElementTree, Element
# import self package library
sys.path.append('/workspace/yo/google_od_api/src/lib/')
from utility import load_config

parser = argparse.ArgumentParser(description='generate txt')
parser.add_argument("--config_path", type = str,
                    default="cfg/txt_generator.json", help="config path")
args = parser.parse_args()
cfg = load_config.readCfg(args.config_path)

def del_file(name:str):
    if os.path.exists(name):
        os.remove(name)
    else:
        print("".join((name, " doens't exists")))

def getIter(xml_obj):
    obj_info = dict()
    for info in xml_obj.iter():
        tag = info.tag
        value = info.text
        if tag != xml_obj.tag:
            obj_info[tag] = value
        else:
            pass
    return obj_info

def readXml(name:str):
    tree = ElementTree()
    tree.parse(name)
    return tree.getroot()

def detection():
    with open(cfg["DETECTION_PATH"], 'r') as json_fd:
        info = json.load(json_fd)
        for name in list(info['imgs'].keys()):
            file_name = "".join((cfg["OUTPUT_PATH"], '/detections/',name.split('.')[0],'.txt'))
            del_file(file_name)
            for i in range(len(info['imgs'][name]['objects'])):
                category = info['imgs'][name]['objects'][i]['category']
                xmin = str(int(info['imgs'][name]['objects'][i]['bbox']['xmin']))
                xmax = str(int(info['imgs'][name]['objects'][i]['bbox']['xmax']))
                ymin = str(int(info['imgs'][name]['objects'][i]['bbox']['ymin']))
                ymax = str(int(info['imgs'][name]['objects'][i]['bbox']['ymax']))
                score = str(info['imgs'][name]['objects'][i]['score'])
                with open(file_name, "a") as predict_fd:
                    string = " ".join((category, score, xmin, ymax, xmax, ymin,'\n'))
                    predict_fd.write(string)

def label():
    for index, image_path in enumerate(glob.glob(cfg["DIR_IMAGE"]+'/*.jpg')):
        name = image_path.rstrip().split('/')[-1]
        file_name = "".join((cfg["OUTPUT_PATH"], '/groundtruths/',image_id,'.txt'))
        del_file(file_name)
        with open("".join((cfg["ANNOTATIONS_PATH"], name, '.xml')), 'r') as label_fd:
            xml_root = readXml(label_fd)
            for m in xml_root.findall('object'):
                category = m.find('name').text
                bd_obj = m.find('bndbox')
                bd_info = getIter(bd_obj)
                with open(file_name, "a") as label_fd:
                    string = " ".join((category, bd_info['xmin'],
                        bd_info['ymax'], bd_info['xmax'], bd_info['ymin'],'\n'))
                    label_fd.write(string)

def mkdir(*directories):
    for directory in list(directories):
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            pass

def main():
    mkdir("".join((cfg["OUTPUT_PATH"], '/groundtruths/')),
          "".join((cfg["OUTPUT_PATH"], '/detections/')))
    label()
    detection()

if __name__ == "__main__":
    main()
