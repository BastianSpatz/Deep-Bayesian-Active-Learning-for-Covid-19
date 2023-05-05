import cv2
import numpy as np

"""
Helper function to exclude visual noise in the images
"""


def find_contours(binary_image):
    """Helper function for finding contours"""
    return cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]


def body_contour(binary_image):
    """Helper function to get body contour"""
    contours = find_contours(binary_image)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    body_idx = np.argmax(areas)
    return contours[body_idx]


def exterior_exclusion(image):
    """Removes visual features exterior to the patient's body"""
    # Create initial binary image
    filt_image = cv2.GaussianBlur(image, (5, 5), 0)
    filt_image.shape = image.shape  # ensure channel dimension is preserved if present
    thresh = cv2.threshold(
        filt_image[filt_image > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[0]
    bin_image = filt_image > thresh

    # Find body contour
    body_cont = body_contour(bin_image.astype(np.uint8))

    # Exclude external regions by replacing with bg mean
    body_mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(body_mask, [body_cont], 0, 1, -1)
    body_mask = body_mask.astype(bool)
    bg_mask = (~body_mask) & (image > 0)
    bg_dark = bg_mask & (~bin_image)  # exclude bright regions from mean
    bg_mean = np.mean(image[bg_dark])
    image[bg_mask] = bg_mean
    return image
