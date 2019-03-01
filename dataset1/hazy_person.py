import os
import cv2
import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom
import glob
import random
import copy

import threading
import queue
import ctypes
import inspect

import train_utils.tools as train_tools
import config

pic_train_dir_str = "PICTURES_LABELS_TRAIN/PICTURES/"
label_train_dir_str = "PICTURES_LABELS_TRAIN/ANOTATION/"

pic_test_dir_str = "PICTURES_LABELS_TEST/PICTURES/"
label_test_dir_str = "PICTURES_LABELS_TEST/ANOTATION/"

class provider(object):
    """provide multi threads API for reading data
    #### multi thread ####            ## multi thread ##
    ######################    ##      ##################    ##     #######################
    ######################      ##    ##################      ##   #######################
    ###read single data###  #######   ### batch data ###  #######  ### load batch data ###
    ######################      ##    ##################      ##   #######################
    ######################    ##      ##################    ##     #######################

    Example 1:
        dt = provider(batch_size=10, for_what="train")
        for step in range(100):
            imgs, labels, t_bboxes = dt.load_batch()
            ## do sth ##

    Example 2:
        with provider(batch_size=10,for_what="train") as pd:
            imgs, labels, t_bboxes = pd.load_batch()
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

        assert batch_size > 0

        if for_what not in ["train", "test"]:
            raise ValueError('pls ensure for_what must be "train" or "test"')
        else:
            self.__for_what = for_what
            data_root = os.path.dirname(__file__)
            if for_what == "train":
                ## load the imgs file name ##
                match = os.path.join(data_root, pic_train_dir_str + "*.jpg")

                self.__imgs_name = glob.glob(match)
                if len(self.__imgs_name) == 0:
                    raise ValueError("can not found the imgs, pls " +
                                     "check pic_train_dir and ensure img format must be jpeg")
                self.__label_dir = os.path.join(data_root, label_train_dir_str)
            else:
                ## load the imgs file name ##
                match = os.path.join(data_root, pic_test_dir_str + "*.jpg")
                self.__imgs_name = glob.glob(match)
                if len(self.__imgs_name) == 0:
                    raise ValueError("can not found the imgs, pls " +
                                     "check pic_train_dir and ensure img format must be jpeg")
                self.__label_dir = os.path.join(data_root, label_test_dir_str)

        self.__batch_size = batch_size
        self.__start_read_data(batch_size=batch_size)
        self.__start_batch_data(batch_size=batch_size)
        pass

    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type!=None:
            print(exc_type)
            print(exc_val)
            print(exc_tb)
            exit(1)

        # if self.__read_threads!=None or self.__batch_threads!=None:
        #     self.stop_loading()
        #     print("kill all threads...")
        #     exit(0)


    def load_batch(self):
        """get the batch data
        Return:
            if dataset is for training, return imgs, labels, t_bboxes:
                imgs: a list of img with the shape (224,224,3)
                a list of label with the len of batch_size
        """
        batch_data = self.__batch_queue.get()
        return batch_data[0], batch_data[1]

    def stop_loading(self):
        """to kill all threads
        """
        threads = self.__read_threads + self.__batch_threads
        for thread in threads:
            self.__async_raise(thread.ident, SystemExit)
        pass


    def __start_read_data(self, batch_size, thread_num=4, capacity_scalar=1):
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
            thread.setDaemon(True)
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
            thread.setDaemon(True)
            thread.start()
            self.__batch_threads.append(thread)


    def __batch_data(self, batch_size):
        """dequeue the data_queue and batch the data into a batch_queue
        """
        first = True
        batch_container_list = []
        while True:
            for i in range(batch_size):
                data_list = self.__data_queue.get()
                if first:
                    ## init the batch_list ##
                    for i in range(len(data_list)):
                        batch_container_list.append([])
                    first = False

                for batch_container,data_item in zip(batch_container_list,data_list):
                    batch_container.append(data_item)

            ## put the batch data into batch_queue ##
            self.__batch_queue.put(copy.deepcopy(batch_container_list))

            for batch_container in batch_container_list:
                batch_container.clear()


    def __send_data(self):
        """ a single thread which send a data to the data queue
        """
        while True:
            img_name = random.sample(self.__imgs_name, 1)[0]
            filename = os.path.basename(img_name)
            basefile = filename.split(".")[0]

            label_name = os.path.join(self.__label_dir,(basefile+".xml"))

            img, bboxes = self.__read_one_sample(img_name,label_name)
            ## resize img and normalize img and bboxes##
            img, bboxes = train_tools.normalize_data(img, bboxes, config.img_output_size)
            # if self.__for_what == "train":
            #     labels, bboxes = \
            #         train_tools.ground_truth_one_img(corner_bboxes=bboxes,
            #                                          priori_boxes=priori_bboxes,
            #                                          surounding_size=2,top_k=2)
            #
            #     ## put data into data queue ##
            #     self.__data_queue.put([img, labels, bboxes])
            # else:
            #     self.__data_queue.put([img, bboxes])
            self.__data_queue.put([img, bboxes])


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

    def __async_raise(self, tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")


if __name__ == '__main__':
    dt = provider(batch_size=10, for_what="train")

    for step in range(100):
        imgs, labels = dt.load_batch()
        ## do sth ##