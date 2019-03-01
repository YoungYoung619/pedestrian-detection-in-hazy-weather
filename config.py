"""
some configuration set here
"""
import numpy as np

"""this priori bboxes are produced by k-means,
encoded by [height, width]
"""
priori_bboxes = np.array([[158, 65], [65, 79]], dtype=np.float32)


"""output img size, encoded by [height, width]
"""
img_output_size = (224,224)
