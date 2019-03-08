import tensorflow as tf

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_16(inputs, is_training, scope='vgg_16'):
    """vgg16--I add the batch normalization in each layer, change the
        activation func from relu to leaky_relu, and delete the fully
        connection layer.
    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        is_training: bool type, whether or not the model is being trained.
        scope: Optional scope for the variables.

    Returns:
        net: the output of the last layer
        end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.max_pool2d],
                            outputs_collections=end_points_collection,):
            with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(0.0005) ):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training,
                                                    'activation_fn':tf.nn.leaky_relu},
                                scope='conv1')
                net = slim.max_pool2d(net, [2, 2],scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training,
                                                    'activation_fn': tf.nn.leaky_relu},
                                scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training,
                                                    'activation_fn': tf.nn.leaky_relu},
                                scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training,
                                                    'activation_fn': tf.nn.leaky_relu},
                                scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training,
                                                    'activation_fn': tf.nn.leaky_relu},
                                scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    return net, end_points