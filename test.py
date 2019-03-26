#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import config

# y, x = np.mgrid[0: config.grid_cell_size[0], 0:config.grid_cell_size[1]]
# x_center = (x.astype(np.float32) + 0.5) / np.float32(config.grid_cell_size[1])
# y_center = (y.astype(np.float32) + 0.5) / np.float32(config.grid_cell_size[0])
# h_pboxes = config.priori_bboxes[:, 0] / config.img_size[0]  ## shape is (len(config.priori_bboxes),)
# w_pboxes = config.priori_bboxes[:, 1] / config.img_size[1]
# y_c_pboxes = np.expand_dims(y_center, axis=-1)  ## shape is (grid_h, grid_w, 1)
# x_c_pboxes = np.expand_dims(x_center, axis=-1)
#
# y_c_pb = []
# x_c_pb = []
# for i in range(len(config.priori_bboxes)):
#     y_c_pb.append(y_c_pboxes)
#     x_c_pb.append(x_c_pboxes)
# y_c_pb = tf.expand_dims(tf.concat(y_c_pb, axis=-1), axis=0)
# x_c_pb = tf.expand_dims(tf.stack(x_c_pb, axis=-1), axis=0)
#
# with tf.Session() as sess:
#     yy, xx = sess.run([y_c_pb, x_c_pb])
#     pass


import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

from dataset1.hazy_person import provider

import cv2

# pd = provider(batch_size=1, for_what="test")
#
# ia.seed(1)
#
# for i in range(100):
#     img, gt = pd.load_batch()
#     img = np.uint8((img[0]+1.)*255/2)
#     gt = np.array(gt[0])*224
#
#     bboxes = []
#     for bbox in gt:
#         x1 = bbox[1]
#         y1 = bbox[0]
#         x2 = bbox[3]
#         y2 = bbox[2]
#         bboxes.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label="person"))
#
#     bbs = ia.BoundingBoxesOnImage(bboxes, shape=img.shape)
#
#     seq = iaa.SomeOf(3,[
#         iaa.Fliplr(0.5),  # horizontal flips
#         iaa.Crop(percent=(0, 0.2)),  # random crops
#         # Small gaussian blur with random sigma between 0 and 0.5.
#         # But we only blur about 50% of all images.
#         iaa.Sometimes(0.5,
#                       iaa.GaussianBlur(sigma=(0, 0.5))
#                       ),
#         # Strengthen or weaken the contrast in each image.
#         iaa.ContrastNormalization((0.75, 1.5)),
#         # Add gaussian noise.
#         # For 50% of all images, we sample the noise once per pixel.
#         # For the other 50% of all images, we sample the noise per pixel AND
#         # channel. This can change the color (not only brightness) of the
#         # pixels.
#         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
#         # Make some images brighter and some darker.
#         # In 20% of all cases, we sample the multiplier once per channel,
#         # which can end up changing the color of the images.
#         iaa.Multiply((0.8, 1.2), per_channel=0.2),
#         # Apply affine transformations to each image.
#         # Scale/zoom them, translate/move them, rotate them and shear them.
#         iaa.Affine(
#             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#             translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#             rotate=[-25,25]
#         ),
#     ], random_order=True)  # apply augmenters in random order
#
#     seq_det = seq.to_deterministic()
#     image_aug = seq_det.augment_images([img])[0]
#     bbs_aug = seq_det.augment_bounding_boxes([bbs])[0].remove_out_of_image().clip_out_of_image()
#
#     for bbox in  bbs_aug.bounding_boxes:
#         c = np.array([bbox.y1, bbox.x1, bbox.y2, bbox.x2])
#         print(c)
#
#     image_aug = ia.imresize_single_image(image_aug, (512, 512))
#     image = ia.imresize_single_image(img, (512, 512))
#     bbs_rescaled = bbs.on(image)
#     bbs_aug = bbs_aug.on(image_aug)
#
#     image_before = bbs_rescaled.draw_on_image(image, thickness=2)
#     image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])
#
#     cv2.imshow("before", image_before)
#     cv2.imshow("test", image_after)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

# import cv2
#
# fname = '1.png'
# img = cv2.imread(fname)
# # 画矩形框
# cv2.rectangle(img, (212,317), (290,436), (0,255,0), 4)
# # 标注文本
# font = cv2.FONT_HERSHEY_SIMPLEX
# text = '001'
# cv2.putText(img, text, (212, 310), font, 2, (0,0,255), 1)
# cv2.imshow("test", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# import collections
# a = [1,2,3,4]
# d = collections.deque(a)
#
# print(type(None))

#
# import os
# import glob
# import cv2
#
# pic_train_dir_str = "dataset1/PICTURES_LABELS_TEMP_TEST/PICTURES/"
# # pic_train_dir_str = "g:/111/hazePerson/baseOnCNN/test/"
# data_root = os.path.dirname(__file__)
# match = os.path.join(data_root, pic_train_dir_str + "*.jpeg")
# imgs_name = glob.glob(match)
#
# for img_name in imgs_name:
#
#     filename = os.path.basename(img_name)
#     dir = os.path.dirname(img_name)
#     filename = filename.split('.')[0]
#
#     img = cv2.imread(img_name)
#     filename = (dir +"/"+ filename + ".jpg")
#     cv2.imwrite(filename=filename, img=img)
#     pass

pd = provider(for_what="evaluate", whether_aug=False)

while (True):
    img, corner_bboxes_gt, file_name = pd.load_data_eval()
    if file_name != None:
        file_name = file_name.split('.')[0]
        file = open(("./evaluation/ground-truth/"+file_name+".txt"),"w")
        for bbox in corner_bboxes_gt:
            bbox = np.int32(bbox*224)
            string = ("person "+str(bbox[1])+" "+str(bbox[0])+" "+str(bbox[3])+" "+str(bbox[2])+"\n")
            file.write(string)
        file.close()
    else:
        break
