"""
some configuration set here
"""
import numpy as np

"""this priori bboxes are produced by k-means,
encoded by [height, width]
"""
#priori_bboxes = np.array([[158, 65], [65, 79]], dtype=np.float32)
#priori_bboxes = np.array([[158, 65], [65, 79]], dtype=np.float32)*2


"""img size, encoded by [height, width],
could not be rondomly change
"""
#img_size = (224,224)
#img_size = (448,448)

"""
grid cell size, means divide img into 7x7 grid cells
"""
# grid_cell_size = (7, 7)
#grid_cell_size = (14, 14)


"""
choose top-k as positive sample
"""
#top_k = 1
#top_k = 4


"""
search range
"""
#surounding_size = 4
#surounding_size = 8


"""
dataset info
"""
n_data_train = 1000


img_size = (224,224)
priori_bboxes = np.array([[158, 65], [65, 79]], dtype=np.float32)
grid_cell_size = (7, 7)
top_k = 1
surounding_size = 4

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