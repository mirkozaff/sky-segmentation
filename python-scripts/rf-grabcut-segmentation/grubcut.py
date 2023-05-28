import numpy as np
import cv2 as cv


def grabCut(img):
    mask_refined = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # Draw the rectangle containing all the object of interest
    # Everything outside is marked ad background
    h, w, _ = img.shape
    rect = (0, 0, w, h//3)
    cv.grabCut(img,mask_refined,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

    # First segmentation based on the rectangle area
    mask_coarse = np.where((mask_refined==2)|(mask_refined==0),0,1).astype('uint8')

    return mask_coarse