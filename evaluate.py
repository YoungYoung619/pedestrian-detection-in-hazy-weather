"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
evaluation

Author：Team Li
"""
import tensorflow as tf
import numpy as np
import cv2
import os
from time import time

from model.factory import model_factory
from dataset1.hazy_person import provider
import utils.test_tools as test_tools

import config

FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'model_name', 'prioriboxes_mbn',
    'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'attention_module', None,
    'The name of attention module to apply.')

tf.app.flags.DEFINE_string(
    'checkpoint_dir', "./checkpoint/",
    'The path to a checkpoint    from which to fine-tune.')

tf.app.flags.DEFINE_float(
    'select_threshold', 0.3, 'obj score less than it would be filter')

tf.app.flags.DEFINE_float(
    'nms_threshold', 0.6, 'nms threshold')

tf.app.flags.DEFINE_integer(
    'keep_top_k', 30, 'maximun num of obj after nms')

tf.app.flags.DEFINE_integer(
    'compare_img_height', 224, 'the img height when compare with ground truth')

tf.app.flags.DEFINE_integer(
    'compare_img_width', 224, 'the img width when compare with ground truth')

tf.app.flags.DEFINE_integer(
    'vis_img_height', 800, 'the img height when visulize')

tf.app.flags.DEFINE_integer(
    'vis_img_width', 800, 'the img width when visulize')

## define placeholder ##
inputs = tf.placeholder(tf.float32,
                        shape=(None, config.img_size[0], config.img_size[1], 3))

def build_graph(model_name, attention_module, is_training):
    """build tf graph for predict
    Args:
        model_name: choose a model to build
        attention_module: must be "se_block" or "cbam_block"
        is_training: whether to train or test, here must be False
    Return:
        det_loss: a tensor with a shape [bs, priori_boxes_num, 4]
        clf_loss: a tensor with a shape [bs, priori_boxes_num, 2]
    """
    assert is_training == False
    net = model_factory(inputs=inputs, model_name=model_name,
                        attention_module=attention_module, is_training=is_training)
    corner_bboxes, clf_pred = net.get_output_for_test()

    score, bboxes = test_tools.bboxes_select(clf_pred, corner_bboxes,
                                             select_threshold= FLAGS.select_threshold)
    score, bboxes = test_tools.bboxes_sort(score, bboxes)
    rscores, rbboxes = test_tools.bboxes_nms_batch(score, bboxes,
                             nms_threshold=FLAGS.nms_threshold,
                             keep_top_k=FLAGS.keep_top_k)
    return rscores, rbboxes

def main(_):
    scores, bboxes = build_graph(FLAGS.model_name, FLAGS.attention_module, is_training=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if FLAGS.checkpoint_dir ==None:
            raise ValueError("checkpoint_dir must not be None")
        else:
            model_name = os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name+".model")
            tf.train.Saver().restore(sess, model_name)
            print("Load checkpoint success...")

        pd = provider(for_what="evaluate", whether_aug=False)

        while (True):
            start = time()
            norm_img, corner_bboxes_gt, file_name = pd.load_data_eval()
            if file_name != None:
                scores_pred, bboxes_pred = sess.run([scores, bboxes], feed_dict={inputs: np.array([norm_img])})
                img = np.uint8((norm_img + 1.) * 255 / 2)
                img = cv2.resize(img, dsize=(FLAGS.vis_img_height, FLAGS.vis_img_width))

                scores_pred = list(scores_pred.values())
                bboxes_pred = list(bboxes_pred.values())
                scores_pred = scores_pred[0][0]
                bboxes_pred = bboxes_pred[0][0]

                bboxes_pred[:, 0] = bboxes_pred[:, 0] * FLAGS.compare_img_height
                bboxes_pred[:, 1] = bboxes_pred[:, 1] * FLAGS.compare_img_width
                bboxes_pred[:, 2] = bboxes_pred[:, 2] * FLAGS.compare_img_height
                bboxes_pred[:, 3] = bboxes_pred[:, 3] * FLAGS.compare_img_width
                bboxes_pred = np.int32(bboxes_pred)

                file_name = file_name.split('.')[0]
                file_name = './evaluation/detection-results/'+file_name+'.txt'
                file = open(file_name, "w")
                for score, bbox in zip(scores_pred, bboxes_pred):
                    if bbox.any() != 0:
                        string = ("person " + str(score) + " " + str(bbox[1]) + " " + str(
                            bbox[0]) + " " + str(bbox[3]) + " " + str(bbox[2]) + "\n")
                        file.write(string)
                file.close()

                # ##vis
                # img = np.uint8((norm_img + 1.) * 255 / 2)
                # imgs_for_gt = cv2.resize(img, dsize=(FLAGS.vis_img_height, FLAGS.vis_img_width))
                # imgs_for_pred = imgs_for_gt.copy()
                # corner_bboxes_gt[:, 0] = corner_bboxes_gt[:, 0] * FLAGS.vis_img_height
                # corner_bboxes_gt[:, 1] = corner_bboxes_gt[:, 1] * FLAGS.vis_img_width
                # corner_bboxes_gt[:, 2] = corner_bboxes_gt[:, 2] * FLAGS.vis_img_height
                # corner_bboxes_gt[:, 3] = corner_bboxes_gt[:, 3] * FLAGS.vis_img_width
                # corner_bboxes_gt = np.int32(corner_bboxes_gt)
                #
                # scores_pred, bboxes_pred = sess.run([scores, bboxes], feed_dict={inputs: np.array([norm_img])})
                #
                # scores_pred = list(scores_pred.values())
                # bboxes_pred = list(bboxes_pred.values())
                # scores_pred = scores_pred[0][0]
                # bboxes_pred = bboxes_pred[0][0]
                #
                # bboxes_pred[:, 0] = bboxes_pred[:, 0] * FLAGS.vis_img_height
                # bboxes_pred[:, 1] = bboxes_pred[:, 1] * FLAGS.vis_img_width
                # bboxes_pred[:, 2] = bboxes_pred[:, 2] * FLAGS.vis_img_height
                # bboxes_pred[:, 3] = bboxes_pred[:, 3] * FLAGS.vis_img_width
                # bboxes_pred = np.int32(bboxes_pred)
                #
                # ## vis ##
                # imgs_for_gt = cv2.cvtColor(imgs_for_gt, cv2.COLOR_BGR2RGB)
                # imgs_for_pred = cv2.cvtColor(imgs_for_pred, cv2.COLOR_BGR2RGB)
                # label = np.ones(corner_bboxes_gt.shape[0], dtype=np.int32)
                # imgs_for_gt = test_tools.visualize_boxes_and_labels_on_image_array(imgs_for_gt, corner_bboxes_gt, label,
                #                                                                    None, config.category_index,
                #                                                                    skip_labels=False)
                #
                # label = np.ones(bboxes_pred.shape[0], dtype=np.int32)
                # imgs_for_pred = test_tools.visualize_boxes_and_labels_on_image_array(imgs_for_pred, bboxes_pred, label,
                #                                                                      scores_pred, config.category_index,
                #                                                                      skip_labels=False)
                # imgs_for_gt = cv2.cvtColor(imgs_for_gt, cv2.COLOR_RGB2BGR)
                # imgs_for_pred = cv2.cvtColor(imgs_for_pred, cv2.COLOR_RGB2BGR)
                #
                # # cv2.imshow("gt", imgs_for_gt)
                # # cv2.imshow("pred", imgs_for_pred)
                # filename = './prediction_imgs/' + file_name
                # cv2.imwrite(filename, imgs_for_pred)
                # # cv2.waitKey(0)
                # # cv2.destroyAllWindows()

                pass
            else:
                tf.logging.info("Finish evaluation")
                break



if __name__ == '__main__':
    tf.app.run()








