from PIL import Image, ImageDraw, ImageFont
from IPython.display import display # to display images
import matplotlib.pyplot as plt
import argparse
import os
import sys
import glob
import cv2
# import self package library start
dirname = (os.path.dirname(os.path.realpath(__file__))).split("/")
dirname[-1] = "lib"
lib_path = "/".join((x for x in dirname))
sys.path.append(lib_path)
from utility import load_config
# import self package library end

parser = argparse.ArgumentParser(description='generator tf_record file')
parser.add_argument("--config_path", type = str,
                    default="cfg/Image_with_bb.json", help="config path")
args = parser.parse_args()
cfg = load_config.readCfg(args.config_path)


ColorTable = dict({'RED': (255, 0, 0),
                  'ORANGE': (255, 165, 0),
                  'YELLOW': (255, 255, 0),
                  'GREEN': (0, 255, 0),
                  'BLUE': (0, 127, 255),
                  'INDIGO': (0, 0, 255),
                  'PURPLE': (139, 0, 255),
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

def drawTextWithMask(draw, loc, text, text_color, mask_color, font):
    ascent, descent = font.getmetrics()
    (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
    left, top, right, bottom = font.getmask(text).getbbox()
    mask_loc = (loc[0], loc[1]+2*bottom, loc[0]+right, loc[1])
    draw.rectangle(mask_loc, mask_color)
    draw.text(loc, text, text_color, font=font)

def drawBoundingBox(draw, file, font):
    if os.path.exists(file):
        with open(file, 'r') as predict_fd:
            for line in predict_fd:
                line = line.rstrip().split(' ')
                if cfg["USAGE"]=='LABELS':
                    text_info = line[0]
                    bb_str = line[1:5]
                    bb = [int(i) for i in bb_str]
                    bb = tuple(bb)
                    draw.rectangle(bb, outline  = ClassColor[cfg["CLASS_COLOR"]][text_info], width=cfg["BOUNDINGBOX_WIDTH"])
                    drawTextWithMask(draw, bb[0:2], ''.join((text_info)), ClassColor[cfg["CLASS_COLOR"]][text_info],
                                     ColorTable['BLACK'], font=font)
                else:
                    text_info = line[0:2]
                    bb_str = line[2:6]
                    bb = [int(i) for i in bb_str]
                    bb = tuple(bb)
                    if float(text_info[1]) > cfg["THRESHOLD"]:
                        draw.rectangle(bb, outline  =  ClassColor[cfg["CLASS_COLOR"]][text_info[0]], width=cfg["BOUNDINGBOX_WIDTH"])
                        drawTextWithMask(draw, bb[0:2], ''.join((text_info[0])), ClassColor[cfg["CLASS_COLOR"]][text_info[0]],
                                         ColorTable['BLACK'], font=font)

def readFileTotalLines(file_path):
    count = 0
    thefile = open(file_path, 'r')
    while 1:
        buffer = thefile.read(8192*1024)
        if not buffer: break
        count += buffer.count('\n')
    thefile.close(  )
    return count

def saveImg(img, filename, index, save_path):
    img.save(''.join((save_path, filename, cfg["SAVE_FILENAME_EXTENSION"])))
    print("index: ", index+1)
    
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass
    
def usageSwitch():
    if cfg["USAGE"]=='PREDICTS':
        boundingbox_path = ''.join((cfg["DETECTIONS_PATH"], '/detections/'))
        save_path = ''.join((cfg["SAVE_PATH"], 'threshoid_', str(cfg["THRESHOLD"]), '/'))
    elif cfg["USAGE"]=='LABELS':
        boundingbox_path = ''.join((cfg["DETECTIONS_PATH"], '/groundtruths/'))
        save_path = ''.join((cfg["SAVE_PATH"], 'groundtruths/'))
    else:
        print('ERROR COMMAND!!')
        boundingbox_path = None
        save_path = None
    return boundingbox_path, save_path

def main():
    font = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Black.ttf", size=28, encoding="unic")
    bounding_box_path, save_path = usageSwitch()
    mkdir(save_path)
    for index, image_path in enumerate(glob.glob(bounding_box_path+'/*.txt')):
        filename = image_path.rstrip().split('/')[-1].split('.')[0]
        print(filename)
        img_name = ''.join((cfg["IMAGE_PATH"], "/", filename,'.jpg'))
        img = Image.open(img_name)
        width, height = img.size
        draw = ImageDraw.Draw(img)
        drawBoundingBox(draw, ''.join((bounding_box_path, filename,'.txt')), font)
        if cfg["DISPLAY"] == 'SHOW':
            display(img)
        elif cfg["DISPLAY"] == 'SAVE':
            saveImg(img, filename, index, save_path)
        else:
            print('ERROR COMMAND!!')
            break
        if index+1 >= cfg["LIMIT_NUM"]:
            break
    print("success")
    
main()