import cv2
import numpy as np


def imdownscale(state, target_shape=(11, 8), max_pix_value=8):
    if state.shape[::-1] == target_shape:
        resized = state
    else:
        resized = cv2.resize(state, target_shape, interpolation=cv2.INTER_AREA)
    img = ((resized / 255.0) * max_pix_value).astype(np.uint8)
    return img
