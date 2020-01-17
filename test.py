"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import cv2
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
import os, glob

pic_train_dir_str = "hazy_person/PICTURES_LABELS_TRAIN/PICTURES/"
label_train_dir_str = "dataset/hazy_person/PICTURES_LABELS_TEMP_TEST/ANOTATION/"


def read_one_sample(label_name):
    """read one sample
    Args:
        img_name: img name, like "/usr/img/image001.jpg"
        label_name: the label file responding the img_name, like "/usr/label/image001.xml"
    Return:
        An ndarray with the shape [img_h, img_w, img_c], bgr format
        An ndarray with the shape [?,4], which means [ymin, xmin, ymax, xmax]
    """
    # cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)


    DOMTree = xml.dom.minidom.parse(label_name)
    collection = DOMTree.documentElement

    objs = collection.getElementsByTagName("object")
    labels = []
    for obj in objs:
        obj_type = obj.getElementsByTagName('name')[0].childNodes[0].data
        if obj_type == "person":
            bbox = obj.getElementsByTagName('bndbox')[0]
            ymin = bbox.getElementsByTagName('ymin')[0].childNodes[0].data
            xmin = bbox.getElementsByTagName('xmin')[0].childNodes[0].data
            ymax = bbox.getElementsByTagName('ymax')[0].childNodes[0].data
            xmax = bbox.getElementsByTagName('xmax')[0].childNodes[0].data
            label = np.array([int(ymin), int(xmin), int(ymax), int(xmax)])
            labels.append(label)
    labels = np.stack(labels, axis=0)
    return labels

xml_files = glob.glob(os.path.join(label_train_dir_str, '*.xml'))
num_img = 0
for xml_file in xml_files:
    l = read_one_sample(xml_file)
    if len(l) <= 2:
        num_img += 1

print(num_img/len(xml_files))
pass