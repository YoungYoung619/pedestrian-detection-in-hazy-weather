"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
evaluation, run this script, then cd into './evaluation' and run eval_tools.py

Author：Team Li
"""
import tensorflow as tf
import numpy as np
import os, time

from model.factory import model_factory
from dataset.hazy_person import provider as hazy_person_pd
from dataset.inria_person import provider as inria_person_pd
from dataset.union_person import provider as union_person_pd
import utils.test_tools as test_tools
from utils.logging import logger
import config


FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim

# tf.app.flags.DEFINE_string(
#     'dataset_name', 'inria_person',
#     'The name of the dataset to train, can be hazy_person, inria_person')

tf.app.flags.DEFINE_string(
    'model_name', 'prioriboxes_mbn',
    'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'attention_module', 'se_block',
    'The name of attention module to apply.')

tf.app.flags.DEFINE_string(
    'checkpoint_dir', './checkpoint/hazy',
    'The path to a checkpoint from which to fine-tune.')

#
# tf.app.flags.DEFINE_string(
#     'model_name', 'prioriboxes_vgg',
#     'The name of the architecture to train.')
#
# tf.app.flags.DEFINE_string(
#     'attention_module', None,
#     'The name of attention module to apply.')
#
# tf.app.flags.DEFINE_string(
#     'checkpoint_dir', './checkpoint/union/pbvgg-k2-224/240k',
#     'The path to a checkpoint from which to fine-tune.')


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


#### config only for prioriboxes_mbn ####
tf.app.flags.DEFINE_string(
    'backbone_name', 'mobilenet_v2',
    'support mobilenet_v1 and mobilenet_v2')

tf.app.flags.DEFINE_boolean(
    'multiscale_feats', True,
    'whether merge different scale features')

## define placeholder ##
inputs = tf.placeholder(tf.float32,
                        shape=(None, config.img_size[0], config.img_size[1], 3))


dataset_map = {'hazy_person': hazy_person_pd,
               'inria_person': inria_person_pd,
               'union_person': union_person_pd}

provider = dataset_map[config.dataset_name]

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
    scores, bboxes = build_graph(FLAGS.model_name, FLAGS.attention_module, is_training=False,
                                 config_dict=config_dict)

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    with tf.Session(config=conf) as sess:
        if ckpt:
            logger.info('loading %s...' % str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('Load checkpoint success...')
        else:
            raise ValueError("can not find checkpoint, pls check checkpoint_dir")

        pd = provider(for_what="evaluate", whether_aug=False)

        time_l = []
        while (True):
            norm_img, corner_bboxes_gt, file_name = pd.load_data_eval()
            if file_name != None:
                t0 =time.time()
                scores_pred, bboxes_pred = sess.run([scores, bboxes], feed_dict={inputs: np.array([norm_img])})
                time_l.append(time.time() - t0)
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
                file_name = './evaluation/'+config.dataset_name+'/detection-results/'+file_name+'.txt'
                file = open(file_name, "w")
                for score, bbox in zip(scores_pred, bboxes_pred):
                    if bbox.any() != 0:
                        string = ("person " + str(score) + " " + str(bbox[1]) + " " + str(
                            bbox[0]) + " " + str(bbox[3]) + " " + str(bbox[2]) + "\n")
                        file.write(string)
                file.close()
                pass
            else:
                tf.logging.info("Finish detection results, please cd into './evaluation' and run eval_tools.py")
                break

        avg_time = np.mean(np.array(time_l))
        print('avg time is ', avg_time)


if __name__ == '__main__':
    tf.app.run()








