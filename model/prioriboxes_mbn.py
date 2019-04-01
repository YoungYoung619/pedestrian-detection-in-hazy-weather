import tensorflow as tf
from model.backbone.mobilenet_v2 import mobilenetv2
from model.backbone.mobilenet_v1 import mobilenet_v1
from model.attention_module import *

slim = tf.contrib.slim

extract_feature_names = {'mobilenet_v2':['layer_7', 'layer_14', 'layer_19'],
                         'mobilenet_v1':['Conv2d_4_pointwise','Conv2d_8_pointwise','Conv2d_13_pointwise']}

def prioriboxes_mbn(inputs, attention_module, is_training, config_dict, bboxs_each_cell=2):
    """ the whole model is inspried by yolov2, what makes our model different is that
        our model use mobilenetV2 as backbone, and use different feature map to do a
        merge, and we add attention module to improve the performance.
    Args:
        inputs: a tensor with the shape of [batch_size, h, w, c], default should
                be [bs, 224, 224, 3], you can try different height and width
                with the input_check setting False, some height and width may
                cause error due to I use tf.space_to_depth to merge different features.
        attention_module: can be se_block or cbam_block
        bboxs_each_cell: describe the number of bboxs in each grib cell
        msf: indicate whether merge multi-scalar feature
        is_training: whether to train
    Return:
        det_out: a tensor with the shape[bs, N, 4], means [y_t, x_t, h_t, w_t]
        clf_out: a tensor with the shape[bs, N, 2], means [bg_score, obj_score]
    """
    assert config_dict['backbone'] in ['mobilenet_v2', 'mobilenet_v1']
    if config_dict['backbone'] == 'mobilenet_v2':
        end_points = mobilenetv2(inputs=inputs, is_training=is_training)
    else:
        end_points = mobilenet_v1(inputs=inputs, is_training=is_training)

    with tf.variable_scope('merge_feat'):
        if config_dict['multiscale_feats']:
            h = []
            for ex_feat_name in extract_feature_names[config_dict['backbone']]:
                h.append(end_points[ex_feat_name].get_shape()[1])
            feats = []
            for i,ex_feat_name in enumerate(extract_feature_names[config_dict['backbone']]):
                with slim.arg_scope([slim.conv2d],
                                    weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                    activation_fn=None):
                    ex_feat = end_points[ex_feat_name]
                    if i < len(extract_feature_names[config_dict['backbone']])-1:
                        ex_feat = slim.conv2d(ex_feat, ex_feat.get_shape()[3] // 2, [1, 1])
                    else:
                        ex_feat = slim.conv2d(ex_feat, ex_feat.get_shape()[3] // 4, [1, 1])
                    ex_feat = slim.batch_norm(ex_feat, is_training=is_training, activation_fn=tf.nn.leaky_relu)
                    if i < len(extract_feature_names[config_dict['backbone']])-1:
                        ex_feat = tf.space_to_depth(ex_feat, block_size=h[i]//h[-1])
                    if attention_module != None:
                        ex_feat = attention_module(ex_feat, name="block%d"%(i))
                    feats.append(ex_feat)

            merge_feat = tf.concat(feats, axis=-1) #feature after merge#
            net = merge_feat
        else:
            net = list(end_points.values())[-1]

        ## the additional conv layers##
        with tf.variable_scope("clf_det_layers"):
            # each bboxs need 6 attribute to describe, means [y_t, x_t, h_t, w_t, bg_socre, obj_score]
            conv_channel_config = [512, 256, 128, 64, 32, bboxs_each_cell * 6]
            for channel in conv_channel_config:
                with slim.arg_scope([slim.conv2d],
                                    weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                    activation_fn=None):
                    net = slim.conv2d(net, channel, [1, 1])
                    net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.leaky_relu)
                    net = slim.conv2d(net, channel, [3, 3])
                    net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.leaky_relu)

            net = slim.batch_norm(net, is_training=is_training, activation_fn=None)
            net = tf.reshape(net,shape=[tf.shape(inputs)[0], -1, 6])
            sz = tf.shape(net)
            det_out = tf.slice(net, begin=[0,0,0], size=[sz[0],sz[1],4]) #[y_t, x_t, h_t, w_t]
            clf_out = tf.slice(net, begin=[0,0,4], size=[sz[0],sz[1],2]) #[bg_socre, obj_score]
        return det_out, clf_out


if __name__ == '__main__':
    imgs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    config_dict = {'multiscale_feats':True,
                   'backbone':'mobilenet_v1'}
    a, b = prioriboxes_mbn(inputs=imgs, attention_module=se_block, is_training=True, config_dict=config_dict)
    pass