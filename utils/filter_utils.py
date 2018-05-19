# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO
from matplotlib import pyplot as plt
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

def filter_white(_img, flag_invert=1):
	tmp = cv2.resize(_img, (640,480))
	hsv = cv2.cvtColor(tmp,cv2.COLOR_BGR2HSV)

	lower_hsv_white = np.array([0, 0, 0])
	upper_hsv_white = np.array([255, 255, 200])

	mask_hsv = cv2.inRange(hsv, lower_hsv_white, upper_hsv_white)
	res_hsv = cv2.bitwise_and(tmp, tmp, mask = mask_hsv)

	res = cv2.bitwise_and(tmp, tmp, mask = mask_hsv)

	return res, mask_hsv

def filter_green(_img, flag_invert=1, flip_order=None):
	tmp = cv2.resize(_img, (640,480))
	hsv = cv2.cvtColor(tmp,cv2.COLOR_BGR2HSV)
	yuv = cv2.cvtColor(tmp,cv2.COLOR_BGR2YUV)

	# lower_yuv_green = np.array([39, 124, 106]) # Original
	lower_yuv_green = np.array([0, 0, 0])
	upper_yuv_green = np.array([202, 255, 120])

	lower_hsv_green = np.array([18, 52, 0])
	# upper_hsv_green = np.array([255, 24, 232]) # Original
	upper_hsv_green = np.array([255, 255, 89])

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

def filter_brown(_img, use_test=True):
	tmp = cv2.resize(_img, (640,480))
	hsv = cv2.cvtColor(tmp,cv2.COLOR_BGR2HSV)
	yuv = cv2.cvtColor(tmp,cv2.COLOR_BGR2YUV)


	if use_test == True:
		lower_yuv_brown = np.array([38, 134, 131])
		upper_yuv_brown = np.array([195, 163, 150])

		lower_hsv_brown = np.array([32, 52, 0])
		upper_hsv_brown = np.array([107, 255, 255])
	else:
		lower_yuv_brown = np.array([0, 0, 0]) # Original
		upper_yuv_brown = np.array([164, 126, 126]) # Original

		upper_hsv_brown = np.array([255, 255, 164]) # Original
		lower_hsv_brown = np.array([32, 52, 118]) # Original


	mask_yuv = cv2.inRange(yuv, lower_yuv_brown, upper_yuv_brown)
	if use_test == False:
		_, mask_yuv = cv2.threshold(mask_yuv, 10, 255, cv2.THRESH_BINARY)
	else:
		_, mask_yuv = cv2.threshold(mask_yuv, 10, 255, cv2.THRESH_BINARY_INV)
	res_yuv = cv2.bitwise_and(tmp, tmp, mask = mask_yuv)


	mask_hsv = cv2.inRange(hsv, lower_hsv_brown, upper_hsv_brown)
	if use_test == False:
		_, mask_hsv = cv2.threshold(mask_hsv, 10, 255, cv2.THRESH_BINARY)
	else:
		_, mask_hsv = cv2.threshold(mask_hsv, 10, 255, cv2.THRESH_BINARY)
	res_hsv = cv2.bitwise_and(tmp, tmp, mask = mask_hsv)


	comp_mask = cv2.bitwise_and(mask_yuv,mask_hsv)
	if use_test == False:
		_, comp_mask = cv2.threshold(comp_mask, 10, 255, cv2.THRESH_BINARY)
	else:
		_, comp_mask = cv2.threshold(comp_mask, 10, 255, cv2.THRESH_BINARY)
	res = cv2.bitwise_and(tmp, tmp, mask = comp_mask)

	return res, comp_mask

def filter_custom(_img, verbose=True,plot_histograms=False):
	tmp = cv2.resize(_img, (640,480))
	hsv = cv2.cvtColor(tmp,cv2.COLOR_BGR2HSV)
	yuv = cv2.cvtColor(tmp,cv2.COLOR_BGR2YUV)

	h, w, c = tmp.shape

	rows_yuv = yuv[(31*h)//32:,:]
	rows_hsv = hsv[(31*h)//32:,:]

	hist_yuv = np.sum(rows_yuv, axis=1)//rows_yuv.shape[1]
	imax_yuv = np.argmax(hist_yuv, axis=0)
	imin_yuv = np.argmin(hist_yuv, axis=0)

	hist_hsv = np.sum(rows_hsv, axis=1)//rows_hsv.shape[1]
	imax_hsv = np.argmax(hist_hsv, axis=0)
	imin_hsv = np.argmin(hist_hsv, axis=0)

	# if verbose == True:
	# 	print("	Max Index YUV: " + str(imax_yuv))
	# 	print("	Min Index YUV: " + str(imin_yuv))
	# 	print("	Max Index HSV: " + str(imax_hsv))
	# 	print("	Min Index HSV: " + str(imin_hsv))

	upper_yuv = np.array([int(hist_yuv[imax_yuv[0],0]),int(hist_yuv[imax_yuv[1],1]),int(hist_yuv[imax_yuv[2],2])])
	lower_yuv = np.array([int(hist_yuv[imin_yuv[0],0]),int(hist_yuv[imin_yuv[1],1]),int(hist_yuv[imin_yuv[2],2])])

	upper_yuv = np.array([int(hist_yuv[imin_yuv[0],0]),int(hist_yuv[imin_yuv[1],1]),int(hist_yuv[imin_yuv[2],2])])
	lower_yuv = np.array([0, 0, 0])

	upper_hsv = np.array([int(hist_hsv[imax_hsv[0],0]),int(hist_hsv[imax_hsv[1],1]),int(hist_hsv[imax_hsv[2],2])])
	lower_hsv = np.array([int(hist_hsv[imin_hsv[0],0]),int(hist_hsv[imin_hsv[1],1]),int(hist_hsv[imin_hsv[2],2])])

	upper_hsv = np.array([255, 255, 164])

	if verbose == True:
		print("	Upper YUV: " + str(upper_yuv))
		print("	Lower YUV: " + str(lower_yuv))
		print("	Upper HSV: " + str(upper_hsv))
		print("	Lower HSV: " + str(lower_hsv))

	if plot_histograms == True:
		plt.figure(6)
		plt.clf()
		plt.subplot(1,2,1)
		plt.title("Histogram: Bottom portion of YUV image")
		plt.plot(range(hist_yuv.shape[0]), hist_yuv[:,0])
		plt.plot(range(hist_yuv.shape[0]), hist_yuv[:,1])
		plt.plot(range(hist_yuv.shape[0]), hist_yuv[:,2])
		plt.subplot(1,2,2)
		plt.title("Histogram: Bottom portion of HSV image")
		plt.plot(range(hist_hsv.shape[0]), hist_hsv[:,0])
		plt.plot(range(hist_hsv.shape[0]), hist_hsv[:,1])
		plt.plot(range(hist_hsv.shape[0]), hist_hsv[:,2])


	mask_yuv = cv2.inRange(yuv, lower_yuv, upper_yuv)
	_, mask_yuv = cv2.threshold(mask_yuv, 10, 255, cv2.THRESH_BINARY)
	res_yuv = cv2.bitwise_and(tmp, tmp, mask = mask_yuv)

	mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
	_, mask_hsv = cv2.threshold(mask_hsv, 10, 255, cv2.THRESH_BINARY)
	res_hsv = cv2.bitwise_and(tmp, tmp, mask = mask_hsv)

	cv2.imshow("YUV Mask",mask_yuv)
	cv2.imshow("HSV Mask",mask_hsv)

	# if flag_invert == 0:
	# 	comp_mask = mask_yuv | mask_hsv
	# if flag_invert == 1:
	# 	comp_mask = mask_yuv & mask_hsv

	comp_mask = cv2.bitwise_and(mask_yuv,mask_hsv)
	_, comp_mask = cv2.threshold(comp_mask, 10, 255, cv2.THRESH_BINARY)
	res = cv2.bitwise_and(tmp, tmp, mask = comp_mask)
	cv2.imshow("Resultant", res)
	# return res, comp_mask

def add_green_mask(white_mask):
	_mask = cv2.cvtColor(white_mask,cv2.COLOR_GRAY2BGR)
	h, w, c = _mask.shape
	green_mask = np.zeros((h,w,3), np.uint8)
	green_mask[:,:] = (0,255,0)

	res_mask = cv2.bitwise_and(green_mask, green_mask, mask = white_mask)
	return res_mask

def apply_morph(_img, ks=[5,5], shape=0, flag_open=False, flag_show=True):
	if shape == 0:
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(int(ks[0]),int(ks[1])))
	elif shape == 1:
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(ks[0]),int(ks[1])))
	else:
		print("alternative structures here...")

	blurred = cv2.medianBlur(_img, 7)
	opening = cv2.morphologyEx(blurred,cv2.MORPH_OPEN,kernel)
	closing = cv2.morphologyEx(blurred,cv2.MORPH_CLOSE,kernel)
	if flag_show == True:
		cv2.imshow('Before Morphing',_img)
		cv2.imshow('Blurred',blurred)
		cv2.imshow('opened',opening)
		cv2.imshow('closed',closing)

	if flag_open == True:
		out = opening
	else:
		out = closing
	return out
