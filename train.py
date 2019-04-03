"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
train net

Author：Team Li
"""
import tensorflow as tf
import numpy as np
import os
from time import time

from model.factory import model_factory
from dataset.hazy_person import provider
import config
from utils.logging import logger

# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'model_name', 'prioriboxes_mbn',
    'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'attention_module', 'se_block',
    '''The name of attention module to apply.
    For prioriboxes_mbn, must be "se_block" 
    , "cbam_block" or None; For prioriboxes_vgg, must
    be None
    ''')

tf.app.flags.DEFINE_string(
    'checkpoint_dir', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'train_dir', './checkpoint/',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_string(
    'summary_dir', './summary/',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

tf.app.flags.DEFINE_integer(
    'batch_size', 30, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'f_log_step', 20,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'f_summary_step', 10,
    'The frequency with which the model is saved, in step.')

tf.app.flags.DEFINE_integer(
    'f_save_step', 2000,
    'The frequency with which summaries are saved, in step.')

tf.app.flags.DEFINE_integer(
    'training_step', None,
    'when training step bigger than training_step, training would stop')

#### config only for prioriboxes_mbn ####
tf.app.flags.DEFINE_string(
    'backbone_name', 'mobilenet_v2',
    'support mobilenet_v1 and mobilenet_v2')

tf.app.flags.DEFINE_boolean(
    'multiscale_feats', True,
    'whether combine different scale features')


FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim

## define placeholder ##
inputs = tf.placeholder(tf.float32,
                        shape=(None, config.img_size[0], config.img_size[1], 3))
bboxes_gt = tf.placeholder(tf.float32,
                        shape=(None, config.grid_cell_size[0]*config.grid_cell_size[1]*\
                               len(config.priori_bboxes), 4))
label_gt = tf.placeholder(tf.int32,
                        shape=(None, config.grid_cell_size[0]*config.grid_cell_size[1]*\
                               len(config.priori_bboxes), 1))
global_step = tf.Variable(0, trainable=False, name='global_step')


def build_graph(model_name, attention_module, config_dict, is_training):
    """build tf graph
    Args:
        model_name: choose a model to build
        attention_module: must be "se_block" or "cbam_block"
        config_dict: some config for building net.
        is_training: whether to train or test
    Return:
        det_loss: a tensor with a shape [bs, priori_boxes_num, 4]
        clf_loss: a tensor with a shape [bs, priori_boxes_num, 2]
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
        return r

    net = model_factory(inputs=inputs, model_name=model_name,
                        attention_module=attention_module, is_training=is_training, config_dict=config_dict)
    bboxes_pred, logits_pred = net.get_output_for_train()

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
        tf.summary.scalar("max_hard_predition", max_hard_pred)  ## the bigger, the better

        nmask = tf.logical_and(tf.cast(neg_mask, dtype=tf.bool),
                               neg_score < max_hard_pred)
        hard_neg_mask = tf.cast(nmask, tf.float32)

        clf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_pred,
                                                                  labels=tf.reshape(label_gt,[-1]))

        pos_loss = tf.reduce_sum(clf_loss * pos_mask)
        neg_loss = tf.reduce_sum(clf_loss * hard_neg_mask)

        # clf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_pred, labels = tf.reshape(label_gt, [-1]))
        # clf_loss = tf.reduce_sum(clf_loss) #/ FLAGS.batch_size

    with tf.name_scope("det_loss_process"):
        det_loss = tf.reduce_sum(_smooth_l1(tf.reshape((bboxes_pred - bboxes_gt),[-1,4])*tf.expand_dims(pos_mask,axis=-1)))# / FLAGS.batch_size

    return det_loss, pos_loss + neg_loss


def build_optimizer(det_loss, clf_loss, var_list=None):
    """ build total loss, and optimizer.
    Args:
        det_loss: a tensor represents the detection loss
        clf_loss: a tensor represents the classification loss
        var_list: the variable need to be trained.
    Return:
        a train_ops
    """
    with tf.name_scope("optimize"):
        loss = 5*det_loss + 0.5*clf_loss

        # learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
        #                                            2*70000 / FLAGS.batch_size,
        #                                            0.97, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            if var_list == None:
                train_ops = optimizer.minimize(loss, global_step= global_step)
            else:
                train_ops = optimizer.minimize(loss, global_step= global_step, var_list=var_list)

        tf.summary.scalar("det_loss", det_loss)
        tf.summary.scalar("clf_loss", clf_loss)
        #tf.summary.scalar("learning_rate", FLAGS.learning_rate)
        return train_ops


def main(_):
    """ start training
    """
    ## assert ##
    logger.info('Asserting parameters')
    assert (FLAGS.f_log_step > 0 or FLAGS.f_log_step == None)
    assert (FLAGS.f_save_step > 0 or FLAGS.f_save_step == None)
    assert (FLAGS.f_summary_step > 0 or FLAGS.f_summary_step == None)

    ## build graph ##
    logger.info('Building graph, using %s...'%(FLAGS.model_name))
    config_dict = {'multiscale_feats': FLAGS.multiscale_feats,
                   'backbone': FLAGS.backbone_name}
    det_loss, clf_loss = build_graph(model_name=FLAGS.model_name,
                                     attention_module=FLAGS.attention_module, is_training=True,
                                     config_dict=config_dict)
    ## build optimizer ##
    train_ops = build_optimizer(det_loss, clf_loss)

    ## summary ops ##
    merge_ops = tf.summary.merge_all()
    logger.info('Build graph success...')
    logger.info('Total trainable parameters:%s'%
                str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

    ## saver
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ## create a summary writer ##
        summary_dir = os.path.join(FLAGS.summary_dir)
        writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

        if FLAGS.checkpoint_dir ==None:
            sess.run(init)
            logger.info('TF variables init success...')
        else:
            model_name = os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name+'.model')
            tf.train.Saver().restore(sess, model_name)
            logger.info('Load checkpoint success...')

        pd = provider(batch_size=FLAGS.batch_size, for_what='train', whether_aug=True)
        avg_det_loss = 0.
        avg_clf_loss = 0.
        avg_time = 0.
        while(True):
            start = time()
            imgs, labels, t_bboex = pd.load_batch()
            imgs = np.array(imgs)
            labels = np.reshape(np.array(labels), newshape=[FLAGS.batch_size, -1, 1])
            t_bboex = np.reshape(np.array(t_bboex), newshape=[FLAGS.batch_size, -1, 4])
            t_ops, m_ops, current_step, d_loss, c_loss \
                =sess.run([train_ops, merge_ops, global_step, det_loss, clf_loss],
                            feed_dict={inputs: imgs, label_gt: labels, bboxes_gt:t_bboex})
            t = round(time() - start, 3)

            if FLAGS.f_log_step != None:
                ## caculate average loss ##
                step = current_step % FLAGS.f_log_step
                avg_det_loss = (avg_det_loss * step + d_loss) / (step + 1.)
                avg_clf_loss = (avg_clf_loss * step + c_loss) / (step + 1.)
                avg_time = (avg_time * step + t) / (step + 1.)
                if current_step%FLAGS.f_log_step == FLAGS.f_log_step-1:
                    ## print info ##
                    logger.info('Step%s det_loss:%s clf_loss:%s time:%s'%(str(current_step),
                                                                            str(avg_det_loss),
                                                                            str(avg_clf_loss),
                                                                            str(avg_time)))
                    avg_det_loss = 0.
                    avg_clf_loss = 0.

            if FLAGS.f_summary_step != None:
                if current_step%FLAGS.f_summary_step == FLAGS.f_summary_step-1:
                    ## summary ##
                    writer.add_summary(m_ops, current_step)

            if FLAGS.f_save_step != None:
                if current_step%FLAGS.f_save_step == FLAGS.f_save_step-1:
                    ## save model ##
                    logger.info('Saving model...')
                    model_name = os.path.join(FLAGS.train_dir,FLAGS.model_name+'.model')
                    saver.save(sess, model_name)
                    logger.info('Save model sucess...')

            if FLAGS.training_step != None:
                if current_step >= FLAGS.training_step:
                    logger.info('Exit training...')
                    break


if __name__ == '__main__':
    tf.app.run()