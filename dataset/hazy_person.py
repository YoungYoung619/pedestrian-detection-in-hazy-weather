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

import utils.train_tools as train_tools
import utils.test_tools as test_tools
import collections

import config

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pic_train_dir_str = "PICTURES_LABELS_TRAIN/PICTURES/"
label_train_dir_str = "PICTURES_LABELS_TRAIN/ANOTATION/"

# pic_test_dir_str = "PICTURES_LABELS_TEST/PICTURES/"
# label_test_dir_str = "PICTURES_LABELS_TEST/ANOTATION/"

pic_test_dir_str = "PICTURES_LABELS_TEMP_TEST/PICTURES/"
label_test_dir_str = "PICTURES_LABELS_TEMP_TEST/ANOTATION/"

class provider(object):
    """provide multi threads API for reading data
    #### multi thread ####            ## multi thread ##
    ######################    ##      ##################    ##     #######################
    ######################      ##    ##################      ##   #######################
    ###read single data###  #######   ### batch data ###  #######  ### load batch data ###
    ######################      ##    ##################      ##   #######################
    ######################    ##      ##################    ##     #######################

    Example 1:
        dt = provider(batch_size=10, for_what="train", whether_aug=True)
        for step in range(100):
            imgs, labels, t_bboxes = dt.load_batch()
            ## do sth ##

        dt = provider(batch_size=10, for_what="predict", whether_aug=True) ##also use aug to predict
        for step in range(100):
            imgs, corner_bboxes = dt.load_batch()
            ## do sth ##

    Example 2:
        with provider(batch_size=10,for_what="train") as pd:
            imgs, labels, t_bboxes = pd.load_batch()
            ## do sth ##

        with provider(batch_size=10,for_what="test") as pd:
            imgs, corner_bboxes = pd.load_batch()
            ## do sth ##
    """
    __imgs_name = None
    __for_what = None
    __whether_aug =None

    __data_queue = None
    __batch_queue = None
    __read_threads = None
    __batch_threads = None
    __threads_name = None

    def __init__(self, for_what, batch_size=1, whether_aug=False):
        """init
        Args:
            batch_size: the size of a batch
            for_what: indicate train or test, must be "train" or "test"
            whether_aug: whether to augument the img
        """
        ##

        assert batch_size > 0
        assert type(whether_aug) == bool

        self.__whether_aug = whether_aug

        if for_what not in ["train", "predict", "evaluate"]:
            raise ValueError('pls ensure for_what must be "train","predict" or "evaluate"')
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
                                     "check pic_test_dir and ensure img format must be jpeg")
                self.__label_dir = os.path.join(data_root, label_test_dir_str)

        if for_what in ["train", "predict"]:
            ## set batch size and start queue ##
            self.__batch_size = batch_size
            self.__threads_name = []
            self.__start_read_data(batch_size=self.__batch_size)
            self.__start_batch_data(batch_size=self.__batch_size)
            logger.info('Start loading queue for %s'%(for_what))
        else:
            ## for evaluation ##
            assert whether_aug == False ## when evaluate, aug must be false
            self.__imgs_name_que = collections.deque(self.__imgs_name)
            logger.info('For evaluation')


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type!=None:
            if self.__threads_name != None:
                if len(self.__threads_name) > 0:
                    exist_threads = threading.enumerate()
                    exist_threads_name = [exist_thread.name for exist_thread in exist_threads]
                    for thread_name in self.__threads_name:
                        if thread_name not in exist_threads_name:
                            names = str.split(thread_name,"_")
                            if names[0] == "read":
                                restart_thread = threading.Thread(target=self.__send_data)
                                restart_thread.setName(thread_name)
                                restart_thread.setDaemon(True)
                                restart_thread.start()
                                print("restart a down thread")
                                return True
                            elif names[0] == "batch":
                                restart_thread = threading.Thread(target=self.__batch_data,
                                                                  args=(self.__batch_size,))
                                restart_thread.setName(thread_name)
                                restart_thread.setDaemon(True)
                                restart_thread.start()
                                print("restart a down thread")
                                return True

            print(exc_type)
            print(exc_val)
            print(exc_tb)
            exit(1)


    def load_data_eval(self):
        """Traversing the test set in sequence
        Return:
            img: one img with shape (h, w, c), if end, None
            bboxes: shape is (n, 4), if end, None
        """

        try:
            ##get img name
            img_name = self.__imgs_name_que.popleft()

            ##get corresponding anotation name
            filename = os.path.basename(img_name)
            basefile = filename.split(".")[0]
            label_name = os.path.join(self.__label_dir, (basefile + '.xml'))

            img, bboxes = self.__read_one_sample(img_name, label_name)
            img, bboxes = train_tools.normalize_data(img, bboxes, config.img_size)
            return img, bboxes, filename
        except IndexError:
            return None, None, None


    def load_batch(self):
        """get the batch data
        Return:
            if dataset is for training, return imgs, labels, t_bboxes:
                imgs: a list of img, with the shape (h, w, c)
                labels: a list of labels, with the shape (grid_h, grid_w, pboxes_num, 1)
                        0 is background, 1 is object
                t_bboxes: a list of t_bboxes with the shape (grid_h, grid_w, pboxes_num, 4)
            if dataset is for test, return imgs, corner_bboxes
                imgs: a list of img, with the shape (h, w, c)
                corner_bboxes: a list of bboxes, with the shape (?, 4), encoded by [ymin,
                xin, ymax, xmax]
        """
        batch_data = self.__batch_queue.get()
        if self.__for_what == "train":
            ## return imgs, lebels, t_bboxes ##
            return batch_data[0], batch_data[1], batch_data[2]
        else:
            ## return imgs , corner_bboxes ##
            return batch_data[0], batch_data[1]

    def stop_loading(self):
        """to kill all threads
        """
        threads = self.__read_threads + self.__batch_threads
        for thread in threads:
            self.__async_raise(thread.ident, SystemExit)
        pass


    def __start_read_data(self, batch_size, thread_num=4, capacity_scalar=2):
        """ start use multi thread to read data to the queue
        Args:
            thread_num: the number of threads used to read data
            batch_size: the buffer size which used to store the data
        """
        self.__read_threads = []
        maxsize = np.maximum(batch_size*capacity_scalar, 5)
        self.__data_queue = queue.Queue(maxsize=maxsize)

        ## start threads
        for i in range(thread_num):
            thread = threading.Thread(target=self.__send_data)
            thread.setDaemon(True)
            thread.setName("read_thread_id%d"%(i))
            self.__threads_name.append("read_thread_id%d"%(i))
            thread.start()
            self.__read_threads.append(thread)


    def __start_batch_data(self, batch_size, thread_num=4, queue_size=5):
        """ start the threads to batch data into the batch_queue
        Args:
            batch_size: the batch size.
            thread_num: the number of threads
            queue_size: the max batch queue length
        """
        assert queue_size > 0

        self.__batch_threads = []
        self.__batch_queue = queue.Queue(queue_size)

        for i in range(thread_num):
            thread = threading.Thread(target=self.__batch_data, args=(batch_size,))
            thread.setDaemon(True)
            thread.setName("batch_thread_id%d"%(i))
            self.__threads_name.append("batch_thread_id%d"%(i))
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
        priori_bboxes = config.priori_bboxes / config.img_size
        while True:
            img_name = random.sample(self.__imgs_name, 1)[0]
            filename = os.path.basename(img_name)
            basefile = filename.split(".")[0]

            label_name = os.path.join(self.__label_dir,(basefile+".xml"))

            img, bboxes = self.__read_one_sample(img_name,label_name)
            ## resize img and normalize img and bboxes##
            if self.__for_what == "train":
                if self.__whether_aug:
                    img, bboxes = train_tools.img_aug(img, bboxes)
                    if len(bboxes) == 0:
                        ## sometimes aug func will corp no person##
                        #logger.warning("No person img, abandoned...")
                        continue
                img, bboxes = train_tools.normalize_data(img, bboxes, config.img_size)
                labels, bboxes = \
                    train_tools.ground_truth_one_img(corner_bboxes=bboxes,
                                                     priori_boxes=priori_bboxes,
                                                     grid_cell_size=config.grid_cell_size,
                                                     surounding_size=config.surounding_size, top_k=config.top_k)

                ## put data into data queue ##
                self.__data_queue.put([img, labels, bboxes])
            else:
                if self.__whether_aug:
                    img, bboxes = test_tools.img_aug(img, bboxes)
                    if len(bboxes) == 0:
                        ## sometimes aug func will corp no person##
                        #logger.warning("No person img, abandoned...")
                        continue
                img, bboxes = train_tools.normalize_data(img, bboxes, config.img_size)
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