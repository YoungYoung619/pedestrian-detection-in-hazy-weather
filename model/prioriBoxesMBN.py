import tensorflow as tf
from model.backbone.mobilenet_v2 import mobilenetv2
from model.attention_module import *

slim = tf.contrib.slim

extract_feature_names = ['layer_7', 'layer_14', 'layer_19']

def prioriBoxesMBN(inputs, attention_module, bboxs_each_cell=2, is_training=True):
    """ the whole model is inspried by yolov2, what makes our model different
        is that our model use mobilenetV2 as backbone
    Args:
        inputs: a tensor with the shape of [batch_size, h, w, c]
        attention_module: can be se_block or cbam_block
        bboxs_each_cell: describe the number of bboxs in each grib cell
        is_training: whether to train
    Return:
    """
    end_points = mobilenetv2(inputs=inputs)

    with tf.variable_scope('merge_feat'):
        h = []
        for ex_feat_name in extract_feature_names:
            h.append(end_points[ex_feat_name].get_shape()[1])

        feats = []
        for i,ex_feat_name in enumerate(extract_feature_names):
            with slim.arg_scope([slim.conv2d],
                                weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                activation_fn=None):
                ex_feat = end_points[ex_feat_name]
                if i < len(extract_feature_names)-1:
                    ex_feat = slim.conv2d(ex_feat, ex_feat.get_shape()[3] // 2, [1, 1])
                else:
                    ex_feat = slim.conv2d(ex_feat, ex_feat.get_shape()[3] // 4, [1, 1])
                ex_feat = slim.batch_norm(ex_feat,is_training=is_training, activation_fn=tf.nn.leaky_relu)
                if i < len(extract_feature_names)-1:
                    ex_feat = tf.space_to_depth(ex_feat, block_size=h[i]//h[-1])
                ex_feat = attention_module(ex_feat, name="block%d"%(i))
                feats.append(ex_feat)

        merge_feat = tf.concat(feats, axis=-1) #feature after merge#

        ## the additional conv layers##
        with tf.variable_scope("clf_det_layers"):
            conv_channel_config = [512, 256, 128, 64, 32, bboxs_each_cell*5] #each bboxs need 5 attribute to describe#
            net = merge_feat
            for channel in conv_channel_config:
                with slim.arg_scope([slim.conv2d],
                                    weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                    activation_fn=None):
                    net = slim.conv2d(net, channel, [1, 1])
                    net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.leaky_relu)
                    net = slim.conv2d(net, channel, [3, 3])
                    net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.leaky_relu)

            net = tf.reshape(net,shape=[-1,5])
            sz = tf.shape(net)
            det_out = tf.slice(net, begin=[0,0], size=[sz[0],4])
            clf_out = tf.slice(net, begin=[0,4], size=[sz[0],1])
            pass



if __name__ == '__main__':
    imgs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    prioriBoxesMBN(inputs=imgs, attention_module=se_block)




