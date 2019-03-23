import tensorflow as tf

def se_block(input_feature, name, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation block.
    As described in https://arxiv.org/abs/1709.01507.
    Args:
        input_feature: a tensor with any shape.
        name: indicate the varibale scope
    Return:
        a tensor after recalibration
    """

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope("se_" + name):
        channel = input_feature.get_shape()[-1]
        # Global average pooling
        squeeze = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)
        excitation = tf.layers.dense(inputs=squeeze,
                                     units=channel // ratio,
                                     activation=tf.nn.relu,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        excitation = tf.layers.dense(inputs=excitation,
                                     units=channel,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')
        scale = input_feature * excitation
    return scale


def cbam_block(inputs, name, reduction_ratio=0.5):
    """Contains the implementation of CBAM.
       As described in https://arxiv.org/pdf/1807.06521.
       Args:
           input_feature: a tensor with any shape.
           name: indicate the varibale scope
       Return:
           a tensor after recalibration
       """
    with tf.variable_scope("cbam_" + name, reuse=tf.AUTO_REUSE):
        batch_size, hidden_num = tf.shape(inputs)[0], inputs.get_shape().as_list()[3]

        maxpool_channel = tf.reduce_max(tf.reduce_max(inputs, axis=1, keepdims=True), axis=2, keepdims=True)
        avgpool_channel = tf.reduce_mean(tf.reduce_mean(inputs, axis=1, keepdims=True), axis=2, keepdims=True)

        maxpool_channel = tf.layers.Flatten()(maxpool_channel)
        avgpool_channel = tf.layers.Flatten()(avgpool_channel)

        mlp_1_max = tf.layers.dense(inputs=maxpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
                                    reuse=None, activation=tf.nn.relu)
        mlp_2_max = tf.layers.dense(inputs=mlp_1_max, units=hidden_num, name="mlp_2", reuse=None)
        mlp_2_max = tf.reshape(mlp_2_max, [batch_size, 1, 1, hidden_num])

        mlp_1_avg = tf.layers.dense(inputs=avgpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
                                    reuse=True, activation=tf.nn.relu)
        mlp_2_avg = tf.layers.dense(inputs=mlp_1_avg, units=hidden_num, name="mlp_2", reuse=True)
        mlp_2_avg = tf.reshape(mlp_2_avg, [batch_size, 1, 1, hidden_num])

        channel_attention = tf.nn.sigmoid(mlp_2_max + mlp_2_avg)
        channel_refined_feature = inputs * channel_attention

        maxpool_spatial = tf.reduce_max(inputs, axis=3, keepdims=True)
        avgpool_spatial = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_avg_pool_spatial = tf.concat([maxpool_spatial, avgpool_spatial], axis=3)
        conv_layer = tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="same",
                                      activation=None)
        spatial_attention = tf.nn.sigmoid(conv_layer)

        refined_feature = channel_refined_feature * spatial_attention

    return refined_feature