"""
some configuration set here
"""
import numpy as np

dataset_name = 'inria_person'  ##can be 'hazy_person or inria_person'

category_index = {0: {"name": "Background"},
                  1: {"name": "Person"}}


# img_size = (224,224)    ##deafult img size---(h, w)
img_size = (448,448)    ##deafult img size---(h, w)
## for hazy person
# priori_bboxes = np.array([[158, 65], [65, 79]], dtype=np.float32)   ##this priori bboxes are produced by k-means,encoded by [height, width]
# priori_bboxes = np.array([[58, 146], [155, 63], [53, 24]], dtype=np.float32)
# priori_bboxes = np.array([[87, 59], [192, 87], [107, 243], [310, 115], [255, 175], [323, 270]], dtype=np.float32)*(448/416)
# priori_bboxes = np.array([[162, 72], [108, 237], [266, 129], [312, 244]], dtype=np.float32)*(448/416)


## for inria person
priori_bboxes = np.array([[134, 47], [72, 22]], dtype=np.float32)
# priori_bboxes = np.array([[83, 31], [153, 48], [78,113], [76,235], [118, 173], [166, 269]], dtype=np.float32)*(448/416)
# priori_bboxes = np.array([[85, 35], [137, 54], [192,91], [258,159]], dtype=np.float32)*(448/416)

## for union_person
# priori_bboxes = np.array([[73, 32], [139, 87]], dtype=np.float32)


grid_cell_size = (14, 14) ##grid cell size, means divide img into 7x7 grid cells
top_k = 2   ##choose top-k as positive sample
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