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
    'model_name', 'prioriboxes_vgg',
    'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'attention_module', 'cbam_block',
    'The name of attention module to apply.')

tf.app.flags.DEFINE_string(
    'checkpoint_dir', "./checkpoint",
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

    with tf.Session() as sess:
        if FLAGS.checkpoint_dir ==None:
            raise ValueError("checkpoint_dir must not be None")
        else:
            model_name = os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name+".model")
            tf.train.Saver().restore(sess, model_name)
            print("Load checkpoint success...")

        pd = provider(for_what="evaluate", whether_aug=False)

        while (True):
            start = time()
            img, corner_bboxes_gt, file_name = pd.load_data_eval()
            if file_name != None:
                scores_pred, bboxes_pred = sess.run([scores, bboxes], feed_dict={inputs: np.array([img])})
                img = np.uint8((img + 1.) * 255 / 2)
                # img = cv2.resize(img, dsize=(FLAGS.vis_img_height, FLAGS.vis_img_width))

                scores_pred = list(scores_pred.values())
                bboxes_pred = list(bboxes_pred.values())
                scores_pred = scores_pred[0][0]
                bboxes_pred = bboxes_pred[0][0]

                bboxes_pred[:, 0] = bboxes_pred[:, 0] * FLAGS.compare_img_height
                bboxes_pred[:, 1] = bboxes_pred[:, 1] * FLAGS.compare_img_width
                bboxes_pred[:, 2] = bboxes_pred[:, 2] * FLAGS.compare_img_height
                bboxes_pred[:, 3] = bboxes_pred[:, 3] * FLAGS.compare_img_width
                bboxes_pred = np.int32(bboxes_pred)

                # file = open(("./forAP/pred/image_" + str(iter*10+i) + ".txt"), "w")
                # #print(file)
                # for index in range(len(xmin_NMS)):
                #     string = ("person " + str(c_NMS[index]) + " " + str(xmin_NMS[index]) + " " + str(ymin_NMS[index]) + " " + str(xmax_NMS[index]) + " " + str(ymax_NMS[index]) + "\n")
                #     #print(string)
                #     file.write(string)
                # file.close()

                file_name = file_name.split('.')[0]
                file_name = './evaluation/detection-results/'+file_name+'.txt'
                file = open(file_name, "w")
                for score, bbox in zip(scores_pred, bboxes_pred):
                    if bbox.any() != 0:
                        string = ("person " + str(score) + " " + str(bbox[1]) + " " + str(
                            bbox[0]) + " " + str(bbox[3]) + " " + str(bbox[2]) + "\n")
                        file.write(string)
                file.close()


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
                # for bbox in bboxes_pred:
                #     if bbox.any() != 0:
                #         cv2.rectangle(img, (bbox[1],bbox[0]), (bbox[3],bbox[2]),
                #                       thickness=2, color=(0,0,255))
                #
                # cv2.imshow("pred", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                pass
            else:
                tf.logging.info("Finish evaluation")
                break



if __name__ == '__main__':
    tf.app.run()








