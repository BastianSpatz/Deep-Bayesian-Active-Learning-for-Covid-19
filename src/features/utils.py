# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2 import model_zoo
from detectron2.config import get_cfg
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from PIL import Image

from pycocotools import mask



def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def create_annotation_info(annotation_id, image_id, category_id, binary_mask):

    # binary_mask = resize_binary_mask(binary_mask, size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    bounding_box = mask.toBbox(binary_mask_encoded)

    is_crowd = 0
    segmentation = binary_mask_to_polygon(binary_mask, 2)
    if not segmentation:
        return None

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id":category_id,
        "iscrowd": is_crowd,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": 512,
        "height": 512,
    } 

    return annotation_info

def build_config(config_name):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.NAME = config_name
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = (
        2  # This is the real "batch size" commonly known to deep learning people
    )
    cfg.SOLVER.BASE_LR = 0.0003  # pick a good LR
    cfg.SOLVER.MAX_ITER = 400000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.EARLY_STOPPING_ROUNDS = 10
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = "./output/" + cfg.NAME
    cfg.TEST.EVAL_PERIOD = 500

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.SOFTMAXES = False
    cfg.MODEL.ROI_MASK_HEAD.DROPOUT_PROBABILITY = 0.5
    cfg.MODEL.ROI_BOX_HEAD.DROPOUT_PROBABILITY = 0.5
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [512]

    with open("configs" + "/" + cfg.NAME + ".yaml", "w") as file:
        file.write(cfg.dump())




