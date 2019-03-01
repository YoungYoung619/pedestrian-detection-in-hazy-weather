import tensorflow as tf

from model.prioriboxes_mbn import prioriboxes_mbn
from dataset1.hazy_person import provider
import train_utils.tools as train_tools

import config

import cv2
import numpy as np
import time

priori_bboxes = config.priori_bboxes / config.img_output_size

pd = provider(batch_size=20,for_what="train")

for i in range(3):
    data = pd.load_batch()
    print("------------")
    for img in data[0]:
        img = np.uint8((img+1.)*255/2)
        print(np.shape(img))
        time.sleep(0.5)
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
pass