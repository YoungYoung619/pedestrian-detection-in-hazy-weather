from model.prioriboxes_mbn import prioriboxes_mbn
from model.prioriboxes_vgg import prioriboxes_vgg
import model.attention_module as attention

import config
import numpy as np
import tensorflow as tf

model_map = {"prioriboxes_mbn":prioriboxes_mbn,
             "prioriboxes_vgg":prioriboxes_vgg}

slim = tf.contrib.slim

class model_factory(object):
    def __init__(self, model_name, attention_module, inputs, config_dict, is_training):
        """init the model_factory
        Args:
            model_name: must be one of model_map
            attention_module: must be "se_block" or "cbam_block"
            inputs: a tensor with shape [bs, h, w, c]
            is_training: indicate whether to train or test.
        """
        assert model_name in model_map.keys()
        assert attention_module in ["se_block", "cbam_block", None]

        # if model_name == "prioriboxes_mbn" and attention_module==None:
        #     raise ValueError("prioriboxes_mbn must choose attention_module")

        self.model_name = model_name
        if attention_module == "se_block":
            self.attention_module = attention.se_block
        elif attention_module == "cbam_block":
            self.attention_module = attention.cbam_block
        else:
            self.attention_module = None

        if model_name == 'prioriboxes_vgg':
            self.det_out, self.clf_out \
                = model_map[model_name](inputs=inputs, attention_module=self.attention_module,
                                        is_training=is_training)
        elif model_name == 'prioriboxes_mbn':
            self.det_out, self.clf_out \
                = model_map[model_name](inputs=inputs, attention_module=self.attention_module,
                                        is_training=is_training, config_dict=config_dict)
        else:
            raise ValueError('error')

    def get_output_for_train(self):
        """get the nets output
        Return:
            det_out: a tensor with a shape [bs, prioriboxes_num, 4], t_bboxes
            clf_out: a tensor with a shape [bs, prioriboxes_num, 2], without softmax
        """
        return self.det_out, self.clf_out
        pass


    def get_output_for_test(self):
        """get the nets output
                Return:
                    det_out: a tensor with a shape [bs, prioriboxes_num, 4],
                             encoded by [ymin, xmin, ymax, xmax]
                    clf_out: a tensor with a shape [bs, prioriboxes_num, 2], after softmax
                """
        y, x = np.mgrid[0: config.grid_cell_size[0], 0:config.grid_cell_size[1]]
        x_center = (x.astype(np.float32) + 0.5) / np.float32(config.grid_cell_size[1])
        y_center = (y.astype(np.float32) + 0.5) / np.float32(config.grid_cell_size[0])
        h_pboxes = config.priori_bboxes[:, 0] / config.img_size[0]  ## shape is (n_pboxes,)
        w_pboxes = config.priori_bboxes[:, 1] / config.img_size[1]
        y_c_pboxes = np.expand_dims(y_center, axis=-1)  ## shape is (grid_h, grid_w, 1)
        x_c_pboxes = np.expand_dims(x_center, axis=-1)

        shape = tf.shape(self.det_out)
        self.det_out = tf.reshape(self.det_out, shape=[-1, config.grid_cell_size[0],
                                                       config.grid_cell_size[1],
                                                       len(config.priori_bboxes), 4])
        y_c_pb = []
        x_c_pb = []
        for i in range(len(config.priori_bboxes)):
            y_c_pb.append(y_c_pboxes)
            x_c_pb.append(x_c_pboxes)
        ## shape is (1, grid_h. grid_w, n_pboxes) ##
        y_c_pb = tf.expand_dims(tf.concat(y_c_pb, axis=-1), axis=0)
        x_c_pb = tf.expand_dims(tf.concat(x_c_pb, axis=-1), axis=0)

        y_t = self.det_out[:, :, :, :, 0]  ##shape is (bs, grid_h, grid_w, n_pboxes)
        x_t = self.det_out[:, :, :, :, 1]
        h_t = self.det_out[:, :, :, :, 2]
        w_t = self.det_out[:, :, :, :, 3]

        ## center_bboxes encoded by [y_c, x_c, h, w]##
        y_c = y_t * h_pboxes + y_c_pb
        x_c = x_t * w_pboxes + x_c_pb
        h = tf.exp(h_t) * h_pboxes
        w = tf.exp(w_t) * w_pboxes

        ## conner_bboxes encoded by [ymin, xmin, ymax, xmax] ##
        ymin = y_c - h / 2.
        xmin = x_c - w / 2.
        ymax = y_c + h / 2.
        xmax = x_c + w / 2.

        corner_bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        corner_bboxes = tf.reshape(corner_bboxes, shape=[shape[0], -1, 4])
        clf_pred = slim.softmax(self.clf_out)

        return corner_bboxes, clf_pred


if __name__ == '__main__':
    imgs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    net = model_factory(inputs=imgs, model_name="prioriboxes_mbn",
                        attention_module="se_block", is_training=True)
    net.get_output_for_test()