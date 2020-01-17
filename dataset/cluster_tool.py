"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import os, glob
from xml.dom.minidom import parse
import xml.dom.minidom

import config
import numpy as np

from sklearn.cluster import KMeans

datase_name = 'inria_person'

inria_person_info = {'img_dir': 'inria_person/PICTURES_LABELS_TRAIN/PICTURES/',
                     'anotation_dir': 'inria_person/PICTURES_LABELS_TRAIN/ANOTATION/'}

hazy_person_info = {'img_dir': 'hazy_person/PICTURES_LABELS_TRAIN/PICTURES/',
                     'anotation_dir': 'hazy_person/PICTURES_LABELS_TRAIN/ANOTATION/'}

dataset_info_map = {'inria_person': inria_person_info,
                    'hazy_person': hazy_person_info}


annotations_name = glob.glob(os.path.join(dataset_info_map[datase_name]['anotation_dir'], '*xml'))

person_h_w = []
for annotation_name in annotations_name:
    DOMTree = xml.dom.minidom.parse(annotation_name)
    collection = DOMTree.documentElement

    objs = collection.getElementsByTagName("size")
    for obj in objs:
        img_w = int(obj.getElementsByTagName('width')[0].childNodes[0].data)
        img_h = int(obj.getElementsByTagName('height')[0].childNodes[0].data)

    objs = collection.getElementsByTagName("object")
    for obj in objs:
        obj_type = obj.getElementsByTagName('name')[0].childNodes[0].data
        if obj_type == "person":
            bbox = obj.getElementsByTagName('bndbox')[0]
            ymin = int(bbox.getElementsByTagName('ymin')[0].childNodes[0].data)
            xmin = int(bbox.getElementsByTagName('xmin')[0].childNodes[0].data)
            ymax = int(bbox.getElementsByTagName('ymax')[0].childNodes[0].data)
            xmax = int(bbox.getElementsByTagName('xmax')[0].childNodes[0].data)

            std_p_h = int((ymax - ymin)/img_h * config.img_size[0])
            std_p_w = int((xmax - xmin) / img_w * config.img_size[1])
            person_h_w.append(np.array([std_p_h, std_p_w]))
    pass


clf = KMeans(n_clusters=2)
clf.fit(np.array(person_h_w))
print(clf.cluster_centers_)
