import tensorflow as tf

from model.factory import model_factory
from dataset1.hazy_person import provider
import train_utils.tools as train_tools

import config

import cv2
import numpy as np
import time

slim = tf.contrib.slim
# with provider(batch_size=10, for_what="train") as pd:
#     for i in range(1000):
#         imgs, labels, t_bboex = pd.load_batch()
#         for img in imgs:
#             img = np.uint8((img+1.)*255/2)
#             cv2.imshow("test", img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#     pass

## define placeholder ##
inputs = tf.placeholder(tf.float32,
                        shape=(None, config.img_size[0], config.img_size[1], 3))
bboxes_gt = tf.placeholder(tf.float32,
                        shape=(None, config.grid_cell_size[0]*config.grid_cell_size[1]*\
                               len(config.priori_bboxes), 4))
label_gt = tf.placeholder(tf.int32,
                        shape=(None, config.grid_cell_size[0]*config.grid_cell_size[1]*\
                               len(config.priori_bboxes), 1))

def build_graph(model_name, attention_module, is_training):
    """build tf graph
    Args:
        model_name: choose a model to build
        attention_module: must be "se_block" or "cbam_block"
        is_training: whether to train or test
    Return:
        det_loss, clf_loss
    """
    def _smooth_l1(x):
        """Smoothed absolute function. Useful to compute an L1 smooth error.
        Define as:
            x^2 / 2         if abs(x) < 1
            abs(x) - 0.5    if abs(x) > 1
        We use here a differentiable definition using min(x) and abs(x). Clearly
        not optimal, but good enough for our purpose!
        """
        absx = tf.abs(x)
        minx = tf.minimum(absx, 1)
        r = 0.5 * ((absx - 1) * minx + absx) ## smooth_l1
        ## i change smooth_l1, casue i wanne the minimum is 0 ##
        return r + 0.5

    net = model_factory(inputs=inputs, model_name=model_name,
                        attention_module=attention_module, is_training=is_training)
    bboxes_pred, logits_pred = net.get_output_for_loss()

    with tf.name_scope("det_loss_process"):
        det_loss = tf.reduce_sum(_smooth_l1(bboxes_pred - bboxes_gt))

    with tf.name_scope("clf_loss_process"):
        logits_pred = tf.reshape(logits_pred, shape=[-1, 2])
        pred = slim.softmax(logits_pred)

        pos_mask = tf.reshape(label_gt, shape=[-1])
        pos_mask = tf.cast(pos_mask, dtype=tf.float32)

        neg_mask = tf.logical_not(tf.cast(pos_mask, dtype=tf.bool))
        neg_mask = tf.cast(neg_mask, dtype=tf.float32)

        # Hard negative mining...
        neg_score = tf.where(tf.cast(neg_mask, dtype=tf.bool),
                             pred[:,0], 1.- neg_mask)

        # Number of negative entries to select.
        neg_ratio = 5.
        pos_num = tf.reduce_sum(pos_mask)
        max_neg_num = tf.cast(tf.reduce_sum(neg_mask),dtype=tf.int32)
        n_neg = tf.cast(neg_ratio * pos_num, tf.int32) + tf.shape(inputs)[0]
        n_neg = tf.minimum(n_neg, max_neg_num)

        val, idxes = tf.nn.top_k(-neg_score, k=n_neg)
        max_hard_pred = -val[-1]
        tf.summary.scalar("max hard predition", max_hard_pred)  ## the bigger, the better

        nmask = tf.logical_and(tf.cast(neg_mask, dtype=tf.bool),
                               neg_score < max_hard_pred)
        hard_neg_mask = tf.cast(nmask, tf.float32)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_pred,
                                                                      labels=tf.reshape(label_gt,[-1]))

        pos_loss = tf.reduce_sum(loss * pos_mask)
        neg_loss = tf.reduce_sum(loss * hard_neg_mask)

        return det_loss, pos_loss + neg_loss

det_loss, clf_loss = build_graph(model_name="prioriboxes_mbn",
                                 attention_module="se_block", is_training=True)

#
# if __name__ == '__main__':
#     pass