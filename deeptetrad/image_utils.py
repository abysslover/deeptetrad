'''
Created on Mar 6, 2019

@author: vincent
'''

import cv2

def pad_image(an_img, shape):
    # shape = (height, width)
    out_img = an_img.copy()
    height_top = (shape[0] - out_img.shape[0]) >> 1
    height_bottom = shape[0] - out_img.shape[0] - height_top
    width_left = (shape[1] - out_img.shape[1]) >> 1
    width_right = shape[1] - out_img.shape[1] - width_left
    return cv2.copyMakeBorder(out_img, height_top, height_bottom, width_left, width_right, cv2.BORDER_CONSTANT)