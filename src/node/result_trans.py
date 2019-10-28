#!/usr/bin/env python
import json
import os
import glob
from xml.etree.ElementTree import ElementTree, Element

RESULT_PATH = './output_b4_with_momentum/bdd100k_test.json'
IMAGESETS_PATH = '/workspace/datasets/BDD100k/bdd100k/images/100k/test/'
ANNOTATIONS_PATH= '/workspace/datasets/BDD100k/VOC_test/Annotations/'
#RESULT_PATH = './output/result_annos_sim.json'
#IMAGESETS_PATH = '/workspace/datasets/BDD100k/VOC_simple/ImageSets/test.txt'
#ANNOTATIONS_PATH= '/workspace/datasets/BDD100k/VOC_simple/Annotations/'

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
    with open(RESULT_PATH, 'r') as json_fd:
        info = json.load(json_fd)
        for name in list(info['imgs'].keys()):
            file_name = "".join(('./predict_results/detections/',name.split('.')[0],'.txt'))
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
    with open(IMAGESETS_PATH, 'r')as imageset_fd:
        for line in imageset_fd:
            name = line[:-1]
            file_name = "".join(('./predict_results/groundtruths/',name,'.txt'))
            del_file(file_name)
            with open("".join((ANNOTATIONS_PATH, name, '.xml')), 'r') as label_fd:
                xml_root = readXml(label_fd)
                for m in xml_root.findall('object'):
                    category = m.find('name').text
                    bd_obj = m.find('bndbox')
                    bd_info = getIter(bd_obj)
                    with open(file_name, "a") as label_fd:
                        string = " ".join((category, bd_info['xmin'],
                            bd_info['ymax'], bd_info['xmax'], bd_info['ymin'],'\n'))
                        label_fd.write(string)

def mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        pass

def main():
    mkdir('predict_results')
    #label()
    detection()

if __name__ == "__main__":
    main()
