import tensorflow as tf
from model.backbone.vgg16 import vgg_16
from model.attention_module import *

slim = tf.contrib.slim

def prioriboxes_vgg(inputs, attention_module, is_training, bboxs_each_cell=2, input_check=True):
    """ the whole model is inspried by yolov2, what makes our model different
        is that our model use vgg as backbone, and we add attention module
    Args:
        inputs: a tensor with the shape of [batch_size, h, w, c], default should
                be [bs, 224, 224, 3], you can try different height and width
                with the input_check setting False, some height and width may
                cause error due to I use tf.space_to_depth to merge different features.
        attention_module: can be se_block or cbam_block
        bboxs_each_cell: describe the number of bboxs in each grib cell
        input_check: default should be [bs, 224, 224, 3], if not, may be error
                     when use the tf.space_to_depth during merge process
        is_training: whether to train
    Return:
        det_out: a tensor with the shape[bs, N, 4], means [y_t, x_t, h_t, w_t]
        clf_out: a tensor with the shape[bs, N, 2], means [bg_score, obj_score]
    """
    shape = inputs.get_shape()
    if input_check:
        if shape[1]!=224 or shape[2]!=224:
            raise ValueError("inputs' height or width must be 224")

    net, end_points = vgg_16(inputs=inputs, is_training=is_training)

    if attention_module != None:
        net = attention_module(net, name="select")

    ## the additional conv layers##
    with tf.variable_scope("clf_det_layers"):
        # each bboxs need 6 attribute to describe, means [y_t, x_t, h_t, w_t, bg_socre, obj_score]
        conv_channel_config = [180, 90, 30, bboxs_each_cell * 6]
        for channel in conv_channel_config:
            with slim.arg_scope([slim.conv2d],
                                weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                activation_fn=None):
                net = slim.conv2d(net, channel, [1, 1])
                net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.leaky_relu)
                net = slim.conv2d(net, channel, [3, 3])
                net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.leaky_relu)
        net = slim.batch_norm(net, is_training=is_training, activation_fn=None)
        net = tf.reshape(net, shape=[tf.shape(inputs)[0], -1, 6])
        sz = tf.shape(net)
        det_out = tf.slice(net, begin=[0, 0, 0], size=[sz[0], sz[1], 4])  # [y_t, x_t, h_t, w_t]
        clf_out = tf.slice(net, begin=[0, 0, 4], size=[sz[0], sz[1], 2])  # [bg_socre, obj_score]
    return det_out, clf_out


if __name__ == '__main__':
    imgs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    prioriboxes_vgg(inputs=imgs, attention_module=se_block, is_training=True)