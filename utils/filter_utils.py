# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO

import cv2
import numpy as np

def select_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([20,60,60])
    upper = np.array([38,174, 250])
    mask = cv2.inRange(hsv, lower, upper)

    return mask

def select_white(image):
    lower = np.array([170,170,170])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)

    return mask

def filter_green(_img, flag_invert=1, flip_order=None):
	tmp = cv2.resize(_img, (640,480))
	hsv = cv2.cvtColor(tmp,cv2.COLOR_BGR2HSV)
	yuv = cv2.cvtColor(tmp,cv2.COLOR_BGR2YUV)

	lower_yuv_green = np.array([39, 124, 106])
	upper_yuv_green = np.array([202, 255, 141])

	lower_hsv_green = np.array([0, 0, 0])
	upper_hsv_green = np.array([255, 24, 232])

	mask_yuv = cv2.inRange(yuv, lower_yuv_green, upper_yuv_green)
	res_yuv = cv2.bitwise_and(tmp, tmp, mask = mask_yuv)

	mask_hsv = cv2.inRange(hsv, lower_hsv_green, upper_hsv_green)
	res_hsv = cv2.bitwise_and(tmp, tmp, mask = mask_hsv)

	if flag_invert == 0:
		comp_mask = mask_yuv | mask_hsv
	if flag_invert == 1:
		comp_mask = mask_yuv & mask_hsv

	res = cv2.bitwise_and(tmp, tmp, mask = comp_mask)

	return res, comp_mask

def filter_brown(_img, flag_invert=1, flip_order=None):
	tmp = cv2.resize(_img, (640,480))
	hsv = cv2.cvtColor(tmp,cv2.COLOR_BGR2HSV)
	yuv = cv2.cvtColor(tmp,cv2.COLOR_BGR2YUV)

	lower_yuv_brown = np.array([0, 0, 0])
	upper_yuv_brown = np.array([164, 126, 255])

	lower_hsv_brown = np.array([32, 52, 118])
	upper_hsv_brown = np.array([255, 255, 255])

	mask_yuv = cv2.inRange(yuv, lower_yuv_brown, upper_yuv_brown)
	res_yuv = cv2.bitwise_and(tmp, tmp, mask = mask_yuv)

	mask_hsv = cv2.inRange(hsv, lower_hsv_brown, upper_hsv_brown)
	res_hsv = cv2.bitwise_and(tmp, tmp, mask = mask_hsv)

	if flag_invert == 0:
		comp_mask = mask_yuv | mask_hsv
	if flag_invert == 1:
		comp_mask = mask_yuv & mask_hsv

	res = cv2.bitwise_and(tmp, tmp, mask = comp_mask)

	return res, comp_mask

def add_green_mask(white_mask):

	_mask = cv2.cvtColor(white_mask,cv2.COLOR_GRAY2BGR)
	h, w, c = _mask.shape
	green_mask = np.zeros((h,w,3), np.uint8)
	green_mask[:,:] = (0,255,0)

	res_mask = cv2.bitwise_and(green_mask, green_mask, mask = white_mask)
	return res_mask
