import sys
import argparse
import glob
import xml.etree.ElementTree as ET
# import self package library
sys.path.append('/workspace/yo/Tensorboard_object_detection_api/src/lib/')
from utility import load_config

parser = argparse.ArgumentParser(description='generator tf_record file')
parser.add_argument("--config_path", type = str,
                    default="cfg/categories_statistics.json", help="config path")
args = parser.parse_args()
cfg = load_config.readCfg(args.config_path)

def main():
    categories_list = {}
    for index, file_name in enumerate(glob.glob(''.join((cfg["xml_path"], "/*.xml")))):
        img_id = file_name.rstrip().split('/')[-1]
        tree = ET.parse(file_name)
        root = tree.getroot()
        for m in root.findall('object'):
            category = m.find('name').text
            if category not in categories_list:
                categories_list[category] = 1
            else:
                categories_list[category] += 1
        print("index: ", index+1, end="\r")
    print("\n", categories_list)
    
    total = 0
    for i in categories_list:
        total += categories_list[i]
    for i in categories_list:
        print("{}: {:.2f}%".format(i, categories_list[i]/total*100))

if __name__ == "__main__":
    main()