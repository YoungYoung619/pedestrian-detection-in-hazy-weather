from __future__ import division
import numpy as np
import cv2
import math
try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except Exception:
    raise ImportError("Pls install imgaug with (pip install imgaug)")

import config

ia.seed(1)

## a seq of img augumentation ##
data_aug_seq = iaa.SomeOf(3,[
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.2)),  # random crops

        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),

        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),

        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),

        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=[-25,25]
        )
    ], random_order=True)  # apply augmenters in random order


def imgaugboxes_2_corner_bboxes(imgaugboxes):
    """"""
    bboxes = []
    for bbox in imgaugboxes.bounding_boxes:
        bboxes.append(np.array([bbox.y1, bbox.x1, bbox.y2, bbox.x2]))

    return np.array(bboxes)


def img_aug(img, corner_bbox):
    """img augumentation
    Args:
        img: ndarray img with any shape [h, w, c]
        corner_bbox: a list or ndarray of bbox with shape [n, 4],
                     encoded by [ymin, xmin, ymax, xmax]
    Return:
        img: after augumentation
        cornet_bbox: after augumentation
    """

    bboxes = []
    for bbox in corner_bbox:
        x1 = bbox[1]
        y1 = bbox[0]
        x2 = bbox[3]
        y2 = bbox[2]
        bboxes.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label="person"))

    bbs = ia.BoundingBoxesOnImage(bboxes, shape=img.shape)

    seq_det = data_aug_seq.to_deterministic()

    ## augumentation ##
    image_aug = seq_det.augment_images([img])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0].remove_out_of_image().clip_out_of_image()

    bboxes = imgaugboxes_2_corner_bboxes(bbs_aug)

    return image_aug, bboxes


def normalize_data(raw_img, corner_bbox, size=(224,224)):
    """make the raw imgs and raw labels into a standard scalar.
    Args:
        raw_imgs: img with any height and width
        corner_bboxes: label encoded by [ymin, xmin, ymax, xmax]
        size: the output img size, default is (224,224)---(height, width)
    Return:
        norm_imgs: a list of img with the same height and width, and its pixel
                    value is between [-1., 1.]
        norm_corner_bboxes: a list of conrner_bboxes [ymin, xmin, ymax, xmax],
                    and its value is between [0., 1.]
    """
    shape = raw_img.shape
    norm_ymin = corner_bbox[:, 0] / shape[0]
    norm_xmin = corner_bbox[:, 1] / shape[1]
    norm_ymax = corner_bbox[:, 2] / shape[0]
    norm_xmax = corner_bbox[:, 3] / shape[1]
    norm_corner_bbox = np.stack([norm_ymin, norm_xmin, norm_ymax, norm_xmax], axis=-1)
    img = cv2.resize(raw_img, dsize=size)
    img = (2.0 / 255.0) * img - 1.0
    return img, norm_corner_bbox


def ground_truth_one_img(corner_bboxes, priori_boxes, grid_cell_size=(7, 7),
                         surounding_size=2, top_k=40):
    """get the ground truth for loss caculation in one img
    Args:
        corner_bboxes: 2D Array, encoded by [ymin, xmin, ymax, xmax], of which
                        the value should be [0., 1.]
        priori_boxes: 2D Array, desribe the height and width of priori bboxes,
                        of which the value should be [0., 1.]
        grid_cell_size: default is (7,7), no need to change unless the shape of
                        net's output changes.
        surounding_size: the range of positive examples searched by algorithm
        top_k: means we choose top-k ovr boxes to be positive boxes
    Return:
        label: a ndarray with the shape (grid_h, grid_w, pboxes_num, 1), in which
                0 indicates background, 1 indicates object.
        transform_info: a ndarray with the shape (grid_h, grid_w, pboxes_num, 4)
                        represents the t_bboxes
    """

    h_per_cell = 1 / grid_cell_size[0]
    w_per_cell = 1 / grid_cell_size[1]
    center_location_h_index = np.int32((corner_bboxes[:, 0] + corner_bboxes[:, 2]) / (2 * h_per_cell))
    center_location_w_index = np.int32((corner_bboxes[:, 1] + corner_bboxes[:, 3]) / (2 * w_per_cell))
    cell_ground_truth_index = np.int32((center_location_h_index * grid_cell_size[1] + center_location_w_index))

    priori_box_index = []  # with [None, center_location_h_index, center_location_w_index, priorBox_index]
    priori_boxes = priori_boxes.reshape([-1, 2])

    for iter in range(len(cell_ground_truth_index)):
        if center_location_h_index[iter] - surounding_size >= 0:
            min_h_index = center_location_h_index[iter] - surounding_size
        else:
            min_h_index = 0
        if center_location_h_index[iter] + surounding_size <= grid_cell_size[0] - 1:
            max_h_index = center_location_h_index[iter] + surounding_size
        else:
            max_h_index = grid_cell_size[0] - 1
        if center_location_w_index[iter] - surounding_size >= 0:
            min_w_index = center_location_w_index[iter] - surounding_size
        else:
            min_w_index = 0
        if center_location_w_index[iter] + surounding_size <= grid_cell_size[1] - 1:
            max_w_index = center_location_w_index[iter] + surounding_size
        else:
            max_w_index = grid_cell_size[1] - 1

        h_indexes = np.arange(min_h_index, max_h_index + 1, 1)
        w_indexes = np.arange(min_w_index, max_w_index + 1, 1)

        ovr_info = []
        for wIndex in w_indexes:
            for hIndex in h_indexes:
                y_c = hIndex * h_per_cell + h_per_cell / 2
                x_c = wIndex * w_per_cell + w_per_cell / 2
                h_p = priori_boxes[:, 0]
                w_p = priori_boxes[:, 1]

                for i in range(len(priori_boxes)):
                    x_min = x_c - w_p[i] / 2
                    x_max = x_c + w_p[i] / 2
                    y_min = y_c - h_p[i] / 2
                    y_max = y_c + h_p[i] / 2
                    areaP = (x_max - x_min) * (y_max - y_min)
                    areaG = (corner_bboxes[iter, 2] - corner_bboxes[iter, 0]) * (
                            corner_bboxes[iter, 3] - corner_bboxes[iter, 1])
                    # compute the IOU
                    xx1 = np.maximum(x_min, corner_bboxes[iter, 1])
                    yy1 = np.maximum(y_min, corner_bboxes[iter, 0])
                    xx2 = np.minimum(x_max, corner_bboxes[iter, 3])
                    yy2 = np.minimum(y_max, corner_bboxes[iter, 2])

                    w = np.maximum(0.0, xx2 - xx1)
                    h = np.maximum(0.0, yy2 - yy1)
                    inter = w * h
                    ovr = inter / (areaP + areaG - inter)
                    ovr_info.append([hIndex, wIndex, i, ovr])
        ovr_info = np.array(ovr_info)
        ovr_info = np.reshape(ovr_info,(-1,4))
        try:
            inds = np.argsort(ovr_info[:, 3])[::-1]
        except IndexError:
            pass

        num = 0
        for index in inds:
            info = [ovr_info[index][0], ovr_info[index][1], ovr_info[index][2]]
            if info not in priori_box_index:  # to avoid same priorBox match multi groundTruth
                if num >= top_k:  # this value means we choose the top-k ovr box to be the positive box
                    break
                priori_box_index.append([ovr_info[index][0], ovr_info[index][1], ovr_info[index][2]])
                num += 1
    priori_box_index = np.array(priori_box_index, dtype=np.int32)
    priori_box_index = priori_box_index.reshape([-1, top_k, 3])

    label = np.zeros(shape=[grid_cell_size[0], grid_cell_size[1], len(priori_boxes),
                            1], dtype=np.int32)  ##0 is background, 1 is object ##
    transform_info = np.zeros(shape=[grid_cell_size[0], grid_cell_size[1],
                                     len(priori_boxes), 4], dtype=np.float32)  ##

    for ground_truth_index in range(len(priori_box_index)):
        corner_bbox = corner_bboxes[ground_truth_index]
        for index_info in priori_box_index[ground_truth_index]:
            y_c_anchor = index_info[0] * h_per_cell + h_per_cell / 2
            x_c_anchor = index_info[1] * w_per_cell + w_per_cell / 2
            h_anchor = priori_boxes[index_info[2], 0]
            w_anchor = priori_boxes[index_info[2], 1]

            y_c_gt = (corner_bbox[0] + corner_bbox[2]) / 2
            x_c_gt = (corner_bbox[1] + corner_bbox[3]) / 2
            h_gt = corner_bbox[2] - corner_bbox[0]
            w_gt = corner_bbox[3] - corner_bbox[1]

            y_t = (y_c_gt - y_c_anchor) / h_anchor
            x_t = (x_c_gt - x_c_anchor) / w_anchor

            ## sometimes imgaug would return the h_gt or w_wt 0 bboxes,
            ## just abandom it
            try:
                h_t = math.log(h_gt / h_anchor)
                w_t = math.log(w_gt / w_anchor)
            except ValueError:
                ## abandom ##
                transform_info[index_info[0], index_info[1], index_info[2]] = [0., 0., 0., 0.]
                label[index_info[0], index_info[1], index_info[2]] = 0
            else:
                transform_info[index_info[0], index_info[1], index_info[2]] = [y_t, x_t, h_t, w_t]
                label[index_info[0], index_info[1], index_info[2]] = 1
    return label, transform_info
