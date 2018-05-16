# Created by: Hunter Young
# Date: 10/5/17
#
# Script Description:
# 	Script is designed to take various commandline arguments making a very simple
# 	and user-friendly method to take any video file of interest as input and extract
# 	all the possible images available into a seperate folder, in addition to outputting
# 	a .csv file logging any additional useful information that is associated with each image.
#
# Current Recommended Usage: (in terminal)
# 	python parse_video.py -p /path/to/videos/home/directory -v name_of_video.mp4 -o name_of_output_data_file
#
# ASSUMPTIONS:
# 	Assumes that the video file name is formatted like so, "X_X_MMDDYY_TimeOfDay_SysTimeMsecRecordingBegins.mp4"

import cv2
import os
import csv
import argparse
from matplotlib import pyplot as plt
import numpy as np
import utils
import filter_utils as fut

def onChange(pos):
	global img
	global _imgs
	global tmp
	global i
	global lower_hsv
	global upper_hsv
	global lower_yuv
	global upper_yuv

	img = _imgs[i]
	# tmp = np.copy(img)
	tmp = cv2.resize(img, (640,480))

	# get current positions of four trackbars
	hmin = cv2.getTrackbarPos('Hmin','image')
	smin = cv2.getTrackbarPos('Smin','image')
	vmin = cv2.getTrackbarPos('Vmin','image')
	hmax = cv2.getTrackbarPos('Hmax','image')
	smax = cv2.getTrackbarPos('Smax','image')
	vmax = cv2.getTrackbarPos('Vmax','image')
	ymin = cv2.getTrackbarPos('Ymin','image')
	umin = cv2.getTrackbarPos('Umin','image')
	vvmin = cv2.getTrackbarPos('VVmin','image')
	ymax = cv2.getTrackbarPos('Ymax','image')
	umax = cv2.getTrackbarPos('Umax','image')
	vvmax = cv2.getTrackbarPos('VVmax','image')
	flag_yuv_inv = cv2.getTrackbarPos('yuvInv','image')
	flag_hsv_inv = cv2.getTrackbarPos('hsvInv','image')
	flag_final_inv = cv2.getTrackbarPos('finalInv','image')

	gray = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(tmp,cv2.COLOR_BGR2HSV)
	yuv = cv2.cvtColor(tmp,cv2.COLOR_BGR2YUV)

	lower_yuv = np.array([ymin, umin, vvmin])
	upper_yuv = np.array([ymax, umax, vvmax])
	mask_yuv = cv2.inRange(yuv, lower_yuv, upper_yuv)

	if flag_yuv_inv == 0:
		ret, mask_yuv = cv2.threshold(mask_yuv, 10, 255, cv2.THRESH_BINARY_INV)
	if flag_yuv_inv == 1:
		ret, mask_yuv = cv2.threshold(mask_yuv, 10, 255, cv2.THRESH_BINARY)

	res_yuv = cv2.bitwise_and(tmp, tmp, mask = mask_yuv)
	cv2.imshow('white mask',mask_yuv)
	# cv2.imshow('white filtered',res_yuv)

	lower_hsv = np.array([hmin, smin, vmin])
	upper_hsv = np.array([hmax, smax, vmax])
	mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

	if flag_hsv_inv == 0:
		ret, mask_hsv = cv2.threshold(mask_hsv, 10, 255, cv2.THRESH_BINARY_INV)
	if flag_hsv_inv == 1:
		ret, mask_hsv = cv2.threshold(mask_hsv, 10, 255, cv2.THRESH_BINARY)

	res_hsv = cv2.bitwise_and(tmp, tmp, mask = mask_hsv)
	cv2.imshow('color mask',mask_hsv)
	# cv2.imshow('color filtered',res_hsv)

	comp_mask = cv2.bitwise_and(mask_yuv,mask_hsv)
	if flag_final_inv == 0:
		_, comp_mask = cv2.threshold(comp_mask, 10, 255, cv2.THRESH_BINARY_INV)
	if flag_final_inv == 1:
		_, comp_mask = cv2.threshold(comp_mask, 10, 255, cv2.THRESH_BINARY)

	res = cv2.bitwise_and(tmp, tmp, mask = comp_mask)

	cv2.imshow('Composite mask',comp_mask)
	cv2.imshow('Composite filtered',res)


#Run Main
if __name__ == "__main__" :

	# Setup commandline argument(s) structures
	ap = argparse.ArgumentParser(description='Image Segmentation')
	ap.add_argument("--pic", "-p", type=str, default='test', metavar='FILE', help="Name of video file to parse")
	# Store parsed arguments into array of variables
	args = vars(ap.parse_args())

	# Extract stored arguments array into individual variables for later usage in script
	_img = args["pic"]
	_imgs, _img_names = utils.get_images_by_dir(_img)

	# create trackbars for color change
	cv2.namedWindow('image')

	# create trackbars for color change
	cv2.createTrackbar('Hmin','image',32,255,onChange)
	cv2.createTrackbar('Smin','image',52,255,onChange)
	cv2.createTrackbar('Vmin','image',118,255,onChange)
	cv2.createTrackbar('Hmax','image',255,255,onChange)
	cv2.createTrackbar('Smax','image',255,255,onChange)
	cv2.createTrackbar('Vmax','image',110,255,onChange)
	cv2.createTrackbar('Ymin','image',0,255,onChange)
	cv2.createTrackbar('Umin','image',0,255,onChange)
	cv2.createTrackbar('VVmin','image',0,255,onChange)
	cv2.createTrackbar('Ymax','image',164,255,onChange)
	cv2.createTrackbar('Umax','image',126,255,onChange)
	cv2.createTrackbar('VVmax','image',126,255,onChange)
	cv2.createTrackbar('yuvInv','image',0,1,onChange)
	cv2.createTrackbar('hsvInv','image',0,1,onChange)
	cv2.createTrackbar('finalInv','image',0,1,onChange)

	img = _imgs[0]	# Input video file as OpenCV VideoCapture device
	tmp = cv2.resize(img, (640,480))
	i = 0
	data_saved = 0
	n = len(_imgs)
	lower_hsv = []
	upper_hsv = []
	lower_yuv = []
	upper_yuv = []

	data_out = []
	# tmpDataHeader = "image_path, lower_yuv [Y], lower_yuv [U], lower_yuv [V], upper_yuv [Y], upper_yuv [U], upper_yuv [V], lower_hsv [H], lower_hsv [S], lower_hsv [V], upper_hsv [H], upper_hsv [S], upper_hsv [V]"
	# data_out.append(np.asarray([tmpDataHeader]))

	while True:
		# print i
		key = cv2.waitKey(5) & 0xFF
		if key == ord('q'):
			print data_out.shape
			np.savetxt("foo.csv", np.asarray(data_out), delimiter=",")
			break
		if key == ord('s'):
			print("Saving Constants for " + str(_img_names[i]))
			# tmpData = str(_img_names[i] + "," + str(lower_yuv[1]) + "," + str(lower_yuv[2]) + "," + str(lower_yuv[3]) + "," + str(upper_yuv[1]) + "," + str(upper_yuv[2]))
			# tmpData = str(tmpData) + "," + str(upper_yuv[2]) + "," + str(lower_hsv[0]) + "," + str(lower_hsv[1]) + "," + str(lower_hsv[2]) + "," + str(upper_hsv[0])
			# tmpData = str(tmpData) + "," + str(upper_hsv[1]) + "," + str(upper_hsv[2])

			tmpData = np.asarray([ _img_names[i] , lower_yuv, upper_yuv, lower_hsv, upper_hsv ], dtype=np.dtype('a16'))
			print tmpData
			# np.savetxt("foo.csv", data_out, delimiter=",")
			print("	YUV Bounds: " + str(lower_yuv) + " --> " + str(upper_yuv))
			print("	HSV Bounds: " + str(lower_hsv) + " --> " + str(upper_hsv))
			data_out.append(tmpData)
			data_saved += 1
		if key == ord('p'):
			i = i + 1
			if i >= n:
				i = 0

			print('Next Image ------ ' + str(_img_names[i]))
			tmp = cv2.resize(_imgs[i], (640,480))
			fut.filter_custom(tmp)
		if key == ord('o'):
			i = i - 1
			if i < 0:
				i = n - 1
			print('Previous Image ------ ' + str(_img_names[i]))
			tmp = cv2.resize(_imgs[i], (640,480))
			fut.filter_custom(tmp)

		cv2.imshow("image", tmp)
	cv2.destroyAllWindows()
