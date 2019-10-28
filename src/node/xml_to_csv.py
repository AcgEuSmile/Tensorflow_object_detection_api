import os
import sys
import json
import glob
import xml.etree.ElementTree as ET


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

class ObjInfo(object):
    def __init__(self, size, file_name, category, bd_info):
        self.category = category
        self.bd_info = bd_info
        self.size = size
        self.file_name = file_name

    def combineString(self):
        answer = self.file_name+','+self.size['width']+','+self.size['height']+','+self.category+','+self.bd_info['xmin']+','+self.bd_info['ymin']+','+self.bd_info['xmax']+','+self.bd_info['ymax']+'\n'
        return answer

def xmlToCsv(path, target_path):
    tree = ET.parse(path)
    root = tree.getroot()
    file_name = root.find('filename').text
    size = getIter(root.find('size'))
    obj_list = list()
    for m in root.findall('object'):
        category = m.find('name').text
        bd_obj = m.find('bndbox')
        bd_info = getIter(bd_obj)
        obj = ObjInfo(size, file_name, category, bd_info)
        obj_list.append(obj)
    with open(target_path, "a") as csv_fd:
        for obj in obj_list:
            csv_fd.write(obj.combineString())

def readCfg(path="cfg/xml2csv_config.json"):
    try:
        with open(path, 'r') as cfg_fd:
            return json.load(cfg_fd)
    except:
        raise("Config file open failed")

def main():
    cfg = readCfg(sys.argv[1]) if len(sys.argv)>1 else readCfg()
    label_path = cfg["label_path"]
    out_path = cfg["out_path"]
    with open(out_path, "w") as csv_fd:
        string='filename,width,height,class,xmin,ymin,xmax,ymax\n'
        csv_fd.write(string)
    for file_name in glob.glob(label_path+'*.xml'):
        xmlToCsv(file_name, out_path)
    print("Success!!")

main()
