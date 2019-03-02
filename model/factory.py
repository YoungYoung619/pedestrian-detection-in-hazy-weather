from model.prioriboxes_mbn import prioriboxes_mbn
import model.attention_module as attention

model_map = {"prioriboxes_mbn":prioriboxes_mbn}

class model_factory(object):
    def __init__(self, model_name, attention_module, inputs, is_training):
        """init the model_factory
        Args:
            model_name: must be one of model_map
            attention_module: must be "se_block" or "cbam_block"
            inputs: a tensor with shape [bs, h, w, c]
            is_training: indicate whether to train or test.
        """
        assert model_name in model_map.keys()
        assert attention_module in ["se_block", "cbam_block"]

        self.model_name = model_name
        if attention_module == "se_block":
            self.attention_module = attention.se_block
        else:
            self.attention_module = attention.cbam_block

        self.det_out, self.clf_out \
            = model_map[model_name](inputs=inputs, attention_module=self.attention_module,
                                    is_training=is_training)


    def get_output_for_loss(self):
        """get the nets output
        Return:
            det_out: a tensor with a shape [bs, prioriboxes_num, 4]
            clf_out: a tensor with a shape [bs, prioriboxes_num, 2], without softmax
        """
        return self.det_out, self.clf_out
        pass