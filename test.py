import numpy as np

from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.training import coordinator
import tensorflow as tf

from dataset1.hazy_person import provider

with provider(batch_size=10, for_what="train") as pd:
    for i in range(1000):
        imgs, labels, t_bboex = pd.load_batch()

        with tf.Session() as session:
            input_fn = numpy_io.numpy_input_fn(imgs, t_bboex, batch_size=10, shuffle=False, num_epochs=1)
    pass