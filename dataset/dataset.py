import os
import cv2
import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom
import glob
import random
import threading
import queue

import time

pic_train_dir = "./PICTURES_LABELS_TRAIN/PICTURES/"
label_train_dir = "./PICTURES_LABELS_TRAIN/ANOTATION/"

pic_test_dir = "./PICTURES_LABELS_TEST/PICTURES/"
label_test_dir = "./PICTURES_LABELS_TEST/ANOTATION/"

class dataset(object):
    """provide multi threads API for reading data
    #### multi thread ####            ## multi thread ##
    ######################    ##      ##################    ##     #######################
    ######################      ##    ##################      ##   #######################
    ###read single data###  #######   ### batch data ###  #######  ### load batch data ###
    ######################      ##    ##################      ##   #######################
    ######################    ##      ##################    ##     #######################

    Example:
        dt = dataset(batch_size=10, for_what="train")
        for step in range(100):
            imgs, labels = dt.load_batch()
            ## do sth ##
    """
    __imgs_name = None
    __for_what = None

    __data_queue = None
    __batch_queue = None
    __read_threads = None
    __batch_threads = None

    def __init__(self, batch_size, for_what):
        """init
        Args:
            batch_size: the size of a batch
            for_what: indicate train or test, must be "train" or "test"
        """
        ##
        if for_what not in ["train", "test"]:
            raise ValueError('pls ensure for_what must be "train" or "test"')
        else:
            self.__for_what = for_what
            if for_what == "train":
                ## load the imgs file name ##
                match = os.path.join(pic_train_dir, "*.jpg")
                self.__imgs_name = glob.glob(match)
                if len(self.__imgs_name) == 0:
                    raise ValueError("can not found the imgs, pls " +
                                     "check pic_train_dir and ensure img format must be jpeg")
                self.__label_dir = label_train_dir
            else:
                ## load the imgs file name ##
                match = os.path.join(pic_test_dir, "*.jpg")
                self.__imgs_name = glob.glob(match)
                if len(self.__imgs_name) == 0:
                    raise ValueError("can not found the imgs, pls " +
                                     "check pic_train_dir and ensure img format must be jpeg")
                self.__label_dir = label_test_dir

        self.__batch_size = batch_size
        self.__start_read_data(batch_size=batch_size)
        self.__start_batch_data(batch_size=batch_size)
        pass

    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        print(exc_type)
        pass


    def load_batch(self):
        """get the batch data
        Return: a list of img with the len of batch_size
                a list of label with the len of batch_size
        """
        batch_data = self.__batch_queue.get()
        return batch_data[0], batch_data[1]


    def get_all_thread(self):
        """ return all threads
        """
        return self.__read_threads + self.__batch_threads


    def __start_read_data(self, batch_size, thread_num=4, capacity_scalar=5):
        """ start use multi thread to read data to the queue
        Args:
            thread_num: the number of threads used to read data
            batch_size: the buffer size which used to store the data
        """
        self.__read_threads = []
        self.__data_queue = queue.Queue(batch_size*capacity_scalar)

        ## start threads
        for i in range(thread_num):
            thread = threading.Thread(target=self.__send_data)
            thread.start()
            self.__read_threads.append(thread)


    def __start_batch_data(self, batch_size, thread_num=4, capacity_scalar=5):
        """ start the threads to batch data into the batch_queue
        Args:
            batch_size: the batch size.
            thread_num: the number of threads
            capacity_scalar: the whole capacity of batch_queue is
                             capacity_scalar*batch_size
        """
        self.__batch_threads = []
        self.__batch_queue = queue.Queue(batch_size*capacity_scalar)

        for i in range(thread_num):
            thread = threading.Thread(target=self.__batch_data, args=(batch_size,))
            thread.start()
            self.__batch_threads.append(thread)


    def __batch_data(self, batch_size):
        """dequeue the data_queue and batch the data into a batch_queue
        """
        while True:
            batch_img = []
            batch_label = []
            for i in range(batch_size):
                data = self.__data_queue.get()
                batch_img.append(data[0])
                batch_label.append(data[1])

            ## put the batch data into batch_queue ##
            self.__batch_queue.put([batch_img, batch_label])


    def __send_data(self):
        """ a single thread which send a data to the data queue
        """
        while True:
            img_name = random.sample(self.__imgs_name, 1)[0]
            filename = os.path.basename(img_name)
            basefile = filename.split(".")[0]

            label_name = os.path.join(self.__label_dir,(basefile+".xml"))

            img, label = self.__read_one_sample(img_name,label_name)

            ## put data into data queue ##
            self.__data_queue.put([img, label])
            #print("send")


    def __read_one_sample(self, img_name, label_name):
        """read one sample
        Args:
            img_name: img name, like "/usr/img/image001.jpg"
            label_name: the label file responding the img_name, like "/usr/label/image001.xml"
        Return:
            An ndarray with the shape [img_h, img_w, img_c], bgr format
            An ndarray with the shape [?,4], which means [ymin, xmin, ymax, xmax]
        """
        #cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        img = cv2.imread(img_name)

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
        labels = np.stack(labels,axis=0)
        return img, labels



if __name__ == '__main__':
    dt = dataset(batch_size=10, for_what="train")

    for step in range(100):
        imgs, labels = dt.load_batch()
        ## do sth ##