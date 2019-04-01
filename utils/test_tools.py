import tensorflow as tf
import numpy as np
try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except Exception:
    raise ImportError("Pls install imgaug")

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


def bboxes_select(predictions_layer, localizations_layer,
                            select_threshold=None,
                            num_classes=2,
                            ignore_class=0,
                            scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = tf.shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = tf.shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer,
                                         tf.stack([l_shape[0], -1, l_shape[-1]]))

        d_scores = {}
        d_bboxes = {}
        for c in range(0, num_classes):
            if c != ignore_class:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes


def bboxes_sort(scores, bboxes, top_k=20, scope=None):
    """Sort bounding boxes by decreasing order and keep only the top_k.
    If inputs are dictionnaries, assume every key is a different class.
    Assume a batch-type input.

    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      top_k: Top_k boxes to keep.
    Return:
      scores, bboxes: Sorted Tensors/Dictionaries of shape Batch x Top_k x 1|4.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_sort_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_sort(scores[c], bboxes[c], top_k=top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_sort', [scores, bboxes]):
        # Sort scores...
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)

        # Trick to be able to use tf.gather: map for each element in the first dim.
        def fn_gather(bboxes, idxes):
            bb = tf.gather(bboxes, idxes)
            return [bb]
        r = tf.map_fn(lambda x: fn_gather(x[0], x[1]),
                      [bboxes, idxes],
                      dtype=[bboxes.dtype],
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        bboxes = r[0]
        return scores, bboxes


def bboxes_nms_batch(scores, bboxes, nms_threshold=0.5, keep_top_k=200,
                     scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Use only on batched-inputs. Use zero-padding in order to batch output
    results.

    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      scores, bboxes Tensors/Dictionaries, sorted by score.
        Padded with zero if necessary.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_nms_batch_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_nms_batch(scores[c], bboxes[c],
                                        nms_threshold=nms_threshold,
                                        keep_top_k=keep_top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_nms_batch'):
        r = tf.map_fn(lambda x: bboxes_nms(x[0], x[1],
                                           nms_threshold, keep_top_k),
                      (scores, bboxes),
                      dtype=(scores.dtype, bboxes.dtype),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        scores, bboxes = r
        return scores, bboxes


def bboxes_nms(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Should only be used on single-entries. Use batch version otherwise.

    Args:
      scores: N Tensor containing float scores.
      bboxes: N x 4 Tensor containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      classes, scores, bboxes Tensors, sorted by score.
        Padded with zero if necessary.
    """
    with tf.name_scope(scope, 'bboxes_nms_single', [scores, bboxes]):
        # Apply NMS algorithm.
        idxes = tf.image.non_max_suppression(bboxes, scores,
                                             keep_top_k, nms_threshold)
        scores = tf.gather(scores, idxes)
        bboxes = tf.gather(bboxes, idxes)
        # Pad results.
        scores = pad_axis(scores, 0, keep_top_k, axis=0)
        bboxes = pad_axis(bboxes, 0, keep_top_k, axis=0)
        return scores, bboxes


def pad_axis(x, offset, size, axis=0, name=None):
    """Pad a tensor on an axis, with a given offset and output size.
    The tensor is padded with zero (i.e. CONSTANT mode). Note that the if the
    `size` is smaller than existing size + `offset`, the output tensor
    was the latter dimension.

    Args:
      x: Tensor to pad;
      offset: Offset to add on the dimension chosen;
      size: Final size of the dimension.
    Return:
      Padded tensor whose dimension on `axis` is `size`, or greater if
      the input vector was larger.
    """
    with tf.name_scope(name, 'pad_axis'):
        shape = get_shape(x)
        rank = len(shape)
        # Padding description.
        new_size = tf.maximum(size-offset-shape[axis], 0)
        pad1 = tf.stack([0]*axis + [offset] + [0]*(rank-axis-1))
        pad2 = tf.stack([0]*axis + [new_size] + [0]*(rank-axis-1))
        paddings = tf.stack([pad1, pad2], axis=1)
        x = tf.pad(x, paddings, mode='CONSTANT')
        # Reshape, to get fully defined shape if possible.
        # TODO: fix with tf.slice
        shape[axis] = size
        x = tf.reshape(x, tf.stack(shape))
        return x

def get_shape(x, rank=None):
    """Returns the dimensions of a Tensor as list of integers or scale tensors.

    Args:
      x: N-d Tensor;
      rank: Rank of the Tensor. If None, will try to guess it.
    Returns:
      A list of `[d1, d2, ..., dN]` corresponding to the dimensions of the
        input tensor.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


########## for vis ############
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import collections

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=False):
  """Adds a bounding box to an image (numpy array).
  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.
  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))
  return image


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.
  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.
  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.
  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.
  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)
  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
  """Draws keypoints on an image.
  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
    draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                  (keypoint_x + radius, keypoint_y + radius)],
                 outline=color, fill=color)


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
  """Draws keypoints on an image (numpy array).
  Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_keypoints_on_image(image_pil, keypoints, color, radius,
                          use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.2,
    agnostic_mode=False,
    line_thickness=6,
    groundtruth_box_visualization_color='red',
    skip_scores=False,
    skip_labels=False):
  """Overlay labeled boxes on an image with formatted scores and label names.
  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.
  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color
      )
    if instance_boundaries is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_boundaries_map[box],
          color='red',
          alpha=1.0
      )
    draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates)

  return image


if __name__ == '__main__':
    import cv2

    img = np.ones(shape=(224, 224, 3), dtype=np.uint8)*255
    bbox = np.array([[10,10,50,50],[30,40,150,150]])
    score = np.array([0.8,0.3])
    label = np.array([1,2])
    category_index = {0:{"name":"background"},
                      1:{"name":"person"},
                      2:{"name":"vehicle"}}
    img = visualize_boxes_and_labels_on_image_array(img, bbox, label, score, category_index)
    #img = draw_bounding_box_on_image_array(img, 10, 10, 150, 150, color='LightCoral',thickness=2,display_str_list=["person"])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("test", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    pass