import numpy as np
import cv2

def histogram_equalization(img: np.array):
    return cv2.equalizeHist(img)
