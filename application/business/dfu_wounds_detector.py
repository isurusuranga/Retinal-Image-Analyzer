import os
import base64
import mimetypes
import json
import numpy as np

import re
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from algorithm.mrcnn import visualize

from algorithm.common_models.DFUImage import DFUImage

import scipy

def convertImage(imgData1, imgSavePath):
    imgstr = ""
    if(re.search(r'base64,(.*)', str(imgData1))):
        imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    else:
        imgstr = imgData1
    with open(imgSavePath, 'wb') as output:
        output.write(base64.b64decode(imgstr))

def encode_image_as_base64_dataurl(file):
    """ Loads an image from path and returns it as base64 encoded string """

    # The image path can be a URI with a query string.
    # Remove any query string elements, basically everything following
    # a question (?) mark
    encoded_string = ""
    prepend_info = 'data:%s;base64' % mimetypes.guess_type(file)[0]

    with open(file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        image_data_base64 = '%s,%s' % (prepend_info, encoded_string)

    return image_data_base64

from importlib import reload # was constantly changin the visualization, so I decided to reload it instead of notebook
reload(visualize)

def save_img_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, img_save_path=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False

    plt.rcParams['savefig.pad_inches'] = 0
    plt.ioff()
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        #auto_show = True

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = visualize.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = patches.Polygon(verts, facecolor="none", edgecolor=color,linewidth = 3)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.autoscale(tight=True)
    plt.savefig(img_save_path)
    plt.close()

def get_dfu_detected_wound_images(query_img_data, dfuMaskRCNNModel, graph, imgSavePath, woundBoundaryPath, woundRegionPath):
    reload(visualize)
    convertImage(query_img_data, imgSavePath)
    image = scipy.misc.imread(imgSavePath)

    with graph.as_default():
        results = dfuMaskRCNNModel.detect([image], verbose=1)
        r = results[0]
        save_img_instances(image, r['rois'], r['masks'], r['class_ids'],
                          ['BG', 'wound'], r['scores'],show_mask=False, show_bbox=False, img_save_path=woundBoundaryPath)
        save_img_instances(image, r['rois'], r['masks'], r['class_ids'],
                          ['BG', 'wound'], r['scores'],show_mask=True, show_bbox=True, img_save_path=woundRegionPath)

        maskrcnn_img_dataurls_list = []

        wound_boundary_image = DFUImage(encode_image_as_base64_dataurl(woundBoundaryPath), 0)
        wound_region_image = DFUImage(encode_image_as_base64_dataurl(woundRegionPath), 1)

        maskrcnn_img_dataurls_list.append(wound_boundary_image.__dict__)
        maskrcnn_img_dataurls_list.append(wound_region_image.__dict__)

    json_response = json.dumps(maskrcnn_img_dataurls_list)
    os.remove(imgSavePath)
    os.remove(woundBoundaryPath)
    os.remove(woundRegionPath)

    return json_response



