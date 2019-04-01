"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
predition

Author：Team Li
"""
import tensorflow as tf
import numpy as np
import cv2
import os
from time import time

from model.factory import model_factory
from dataset.hazy_person import provider
import utils.test_tools as test_tools
from utils.logging import logger

import config

FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'model_name', None,
    'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'attention_module', None,
    'The name of attention module to apply.')

tf.app.flags.DEFINE_string(
    'checkpoint', None,
    'The full file name to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_float(
    'select_threshold', 0.3, 'obj score less than it would be filter')

tf.app.flags.DEFINE_float(
    'nms_threshold', 0.6, 'nms threshold')

tf.app.flags.DEFINE_integer(
    'keep_top_k', 30, 'maximun num of obj after nms')

tf.app.flags.DEFINE_integer(
    'vis_img_height', 512, 'the img height when visulize')

tf.app.flags.DEFINE_integer(
    'vis_img_width', 512, 'the img width when visulize')

#### config only for prioriboxes_mbn ####
tf.app.flags.DEFINE_string(
    'backbone_name', None,
    'support mobilenet_v1 and mobilenet_v2')

tf.app.flags.DEFINE_boolean(
    'multiscale_feats', None,
    'whether merge different scale features')

tf.app.flags.DEFINE_boolean(
    'whether_aug', None,
    'whether use augmentation to prediction')

## define placeholder ##
inputs = tf.placeholder(tf.float32,
                        shape=(None, config.img_size[0], config.img_size[1], 3))

def build_graph(model_name, attention_module, config_dict, is_training):
    """build tf graph for predict
    Args:
        model_name: choose a model to build
        attention_module: must be "se_block" or "cbam_block"
        config_dict: some config for building net
        is_training: whether to train or test, here must be False
    Return:
        det_loss: a tensor with a shape [bs, priori_boxes_num, 4]
        clf_loss: a tensor with a shape [bs, priori_boxes_num, 2]
    """
    assert is_training == False
    net = model_factory(inputs=inputs, model_name=model_name,
                        attention_module=attention_module, is_training=is_training,
                        config_dict=config_dict)
    corner_bboxes, clf_pred = net.get_output_for_test()

    score, bboxes = test_tools.bboxes_select(clf_pred, corner_bboxes,
                                             select_threshold= FLAGS.select_threshold)
    score, bboxes = test_tools.bboxes_sort(score, bboxes)
    rscores, rbboxes = test_tools.bboxes_nms_batch(score, bboxes,
                             nms_threshold=FLAGS.nms_threshold,
                             keep_top_k=FLAGS.keep_top_k)
    return rscores, rbboxes

def main(_):
    config_dict = {'multiscale_feats': FLAGS.multiscale_feats,
                   'backbone': FLAGS.backbone_name}
    scores, bboxes = build_graph(FLAGS.model_name, FLAGS.attention_module,
                                 config_dict=config_dict, is_training=False)

    configuretion = tf.ConfigProto()
    configuretion.gpu_options.allow_growth = True
    with tf.Session(config=configuretion) as sess:
        if FLAGS.checkpoint ==None:
            raise ValueError("checkpoint_dir must not be None")
        else:
            tf.train.Saver().restore(sess, FLAGS.checkpoint)
            print("Load checkpoint success...")

        pd = provider(batch_size=1, for_what="predict", whether_aug=FLAGS.whether_aug)
        logger.info("Please press any key to skip picture...")
        while (True):
            # start = time()
            # norm_imgs, labels, corner_bboxes_gt = pd.load_batch()
            norm_imgs, corner_bboxes_gt = pd.load_batch()
            #print(corner_bboxes_gt)
            imgs = np.uint8((norm_imgs[0] + 1.)*255 / 2)
            imgs_for_gt = cv2.resize(imgs, dsize=(FLAGS.vis_img_height, FLAGS.vis_img_width))
            imgs_for_pred = imgs_for_gt.copy()
            corner_bboxes_gt = corner_bboxes_gt[0]
            corner_bboxes_gt[:, 0] = corner_bboxes_gt[:, 0] * FLAGS.vis_img_height
            corner_bboxes_gt[:, 1] = corner_bboxes_gt[:, 1] * FLAGS.vis_img_width
            corner_bboxes_gt[:, 2] = corner_bboxes_gt[:, 2] * FLAGS.vis_img_height
            corner_bboxes_gt[:, 3] = corner_bboxes_gt[:, 3] * FLAGS.vis_img_width
            corner_bboxes_gt = np.int32(corner_bboxes_gt)

            scores_pred, bboxes_pred = sess.run([scores, bboxes], feed_dict={inputs:np.array(norm_imgs)})

            scores_pred = list(scores_pred.values())
            bboxes_pred = list(bboxes_pred.values())
            scores_pred = scores_pred[0][0]
            bboxes_pred = bboxes_pred[0][0]

            bboxes_pred[:, 0] = bboxes_pred[:, 0] * FLAGS.vis_img_height
            bboxes_pred[:, 1] = bboxes_pred[:, 1] * FLAGS.vis_img_width
            bboxes_pred[:, 2] = bboxes_pred[:, 2] * FLAGS.vis_img_height
            bboxes_pred[:, 3] = bboxes_pred[:, 3] * FLAGS.vis_img_width
            bboxes_pred = np.int32(bboxes_pred)

            ## vis ##
            imgs_for_gt = cv2.cvtColor(imgs_for_gt, cv2.COLOR_BGR2RGB)
            imgs_for_pred = cv2.cvtColor(imgs_for_pred, cv2.COLOR_BGR2RGB)
            label = np.ones(corner_bboxes_gt.shape[0], dtype=np.int32)
            imgs_for_gt = test_tools.visualize_boxes_and_labels_on_image_array(imgs_for_gt, corner_bboxes_gt, label,
                                                                     None, config.category_index, skip_labels=False)

            label = np.ones(bboxes_pred.shape[0], dtype=np.int32)
            imgs_for_pred = test_tools.visualize_boxes_and_labels_on_image_array(imgs_for_pred, bboxes_pred, label,
                                                                                   scores_pred, config.category_index,
                                                                                   skip_labels=False)
            imgs_for_gt = cv2.cvtColor(imgs_for_gt, cv2.COLOR_RGB2BGR)
            imgs_for_pred = cv2.cvtColor(imgs_for_pred, cv2.COLOR_RGB2BGR)


            cv2.imshow("ground-truth", imgs_for_gt)
            cv2.imshow("prediction", imgs_for_pred)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            pass


if __name__ == '__main__':
    tf.app.run()








