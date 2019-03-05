#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import config

y, x = np.mgrid[0: config.grid_cell_size[0], 0:config.grid_cell_size[1]]
x_center = (x.astype(np.float32) + 0.5) / np.float32(config.grid_cell_size[1])
y_center = (y.astype(np.float32) + 0.5) / np.float32(config.grid_cell_size[0])
h_pboxes = config.priori_bboxes[:, 0] / config.img_size[0]  ## shape is (len(config.priori_bboxes),)
w_pboxes = config.priori_bboxes[:, 1] / config.img_size[1]
y_c_pboxes = np.expand_dims(y_center, axis=-1)  ## shape is (grid_h, grid_w, 1)
x_c_pboxes = np.expand_dims(x_center, axis=-1)

y_c_pb = []
x_c_pb = []
for i in range(len(config.priori_bboxes)):
    y_c_pb.append(y_c_pboxes)
    x_c_pb.append(x_c_pboxes)
y_c_pb = tf.expand_dims(tf.concat(y_c_pb, axis=-1), axis=0)
x_c_pb = tf.expand_dims(tf.stack(x_c_pb, axis=-1), axis=0)

with tf.Session() as sess:
    yy, xx = sess.run([y_c_pb, x_c_pb])
    pass