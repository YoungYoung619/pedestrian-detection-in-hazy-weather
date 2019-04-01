"""
some configuration set here
"""
import numpy as np


category_index = {0: {"name": "Background"},
                  1: {"name": "Person"}}


img_size = (224,224)    ##deafult img size---(h, w)
priori_bboxes = np.array([[158, 65], [65, 79]], dtype=np.float32)   ##this priori bboxes are produced by k-means,encoded by [height, width]
grid_cell_size = (7, 7) ##grid cell size, means divide img into 7x7 grid cells
top_k = 1   ##choose top-k as positive sample
surounding_size = 4     ##surounding_size = 4

# img_size = (448,448)
# priori_bboxes = np.array([[158, 65], [65, 79]], dtype=np.float32)*2
# grid_cell_size = (14, 14)
# top_k = 2
# surounding_size = 5

# img_size = (320,320)
# priori_bboxes = np.array([[158, 65], [65, 79]], dtype=np.float32)*1.4285
# grid_cell_size = (10, 10)
# top_k = 1
# surounding_size = 5