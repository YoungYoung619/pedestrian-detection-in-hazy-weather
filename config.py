"""
some configuration set here
"""
import numpy as np

"""this priori bboxes are produced by k-means,
encoded by [height, width]
"""
priori_bboxes = np.array([[158, 65], [65, 79]], dtype=np.float32)


"""img size, encoded by [height, width]
"""
img_size = (224,224)

"""
grid cell size, means divide img into 7x7 grid cells
"""
grid_cell_size = (7, 7)

"""
dataset info
"""
n_data_train = 1000
