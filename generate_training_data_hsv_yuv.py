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
from utils import utils as ut
from utils import filter_utils as fut
from utils import line_utils as lut
from utils import seg_utils as sut

def onChange(pos):
	global img
	global _imgs
	global tmp
	global i
	global lower_hsv
	global upper_hsv
	global lower_yuv
	global upper_yuv

	xbuf = 100
	img = _imgs[i]
	# tmp = np.copy(img)
	tmp = cv2.resize(img, (640,480))
	h,w,c = tmp.shape

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
	ks1 = cv2.getTrackbarPos('ks1','image')
	ks2 = cv2.getTrackbarPos('ks2','image')
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

	hsv = cv2.cvtColor(tmp,cv2.COLOR_BGR2HSV)

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

	tmpFil = fut.apply_morph(res, ks=[ks1,ks2],flag_open=True)
	ret, threshed_img = cv2.threshold(cv2.cvtColor(tmpFil, cv2.COLOR_BGR2GRAY),2, 255, cv2.THRESH_BINARY)

	hist = sut.custom_hist(threshed_img,[0,h],[xbuf,w-xbuf],axis=0)
	smoothHist = sut.histogram_sliding_filter(hist, window_size=16)
	xmid = sut.find_row_ends(smoothHist)

	# Crop the image into two halves
	try:
		begL = 0
		endL = int(xmid[0])
		begR = int(xmid[1])
		endR = w
	except:
		begL = 0
		endL = int(xmid)
		begR = endL
		endR = w

	img_left = tmpFil[:,begL:endL]
	img_right = tmpFil[:,begR:endR]

	ret, threshed_imgL = cv2.threshold(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY),10, 255, cv2.THRESH_BINARY)
	ret, threshed_imgR = cv2.threshold(cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY),10, 255, cv2.THRESH_BINARY)

	lineL, lineR, disp_lines,_,_ = lut.ransac_meth2(tmpFil,endL,begR,max_trials=50,stop_probability=0.80, display=tmp,y_offset=0)
	cv2.imshow("morphed",tmpFil)
	cv2.imshow("lines",disp_lines)
	# post_img = lut.updateLines(res)


#Run Main
if __name__ == "__main__" :

	# Setup commandline argument(s) structures
	ap = argparse.ArgumentParser(description='Image Segmentation')
	ap.add_argument("--images_path", "-i", type=str, default='test', metavar='FILE', help="Name of video file to parse")
	ap.add_argument("--output_file", "-n", type=str, metavar='NAME', default='training_log_hsv', help="Name of output file containing information about parsed video")
	ap.add_argument("--output_path", "-p", type=str, metavar='FILE', default='exported', help="Name of video file to parse")
	# Store parsed arguments into array of variables
	args = vars(ap.parse_args())

	# Extract stored arguments array into individual variables for later usage in script
	imgDir = args["images_path"]
	outName = args["output_file"]
	outDir = args["output_path"]

	outName = str(imgDir) + "_" + str(outName)
	# outDir = os.getcwd() + "/" + str(outDir)
	print("	Output Directory:			" + os.getcwd() + "/" + str(outDir))

	_imgs, _img_names = ut.get_images_by_dir(imgDir)

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
	cv2.createTrackbar('ks1','image',3,20,onChange)
	cv2.createTrackbar('ks2','image',12,20,onChange)
	cv2.createTrackbar('yuvInv','image',0,1,onChange)
	cv2.createTrackbar('hsvInv','image',0,1,onChange)
	cv2.createTrackbar('finalInv','image',0,1,onChange)

	img = _imgs[0]	# Input video file as OpenCV VideoCapture device
	tmp = cv2.resize(img, (640,480))
	i = 0; n = len(_imgs); count_record = 0
	lower_hsv = []
	upper_hsv = []
	lower_yuv = []
	upper_yuv = []

	csvHeaders = ["image", "lower_yuv_y", "lower_yuv_u", "lower_yuv_v", "upper_yuv_y", "upper_yuv_u", "upper_yuv_v", "lower_hsv_h", "lower_hsv_s", "lower_hsv_v", "upper_hsv_h", "upper_hsv_s","upper_hsv_v"]
	csvList = []

	while True:
		# print i
		key = cv2.waitKey(5) & 0xFF
		if key == ord('q'):
			# print data_out.shape
			# np.savetxt("foo.csv", np.asarray(data_out), delimiter=",")
			break
		if key == ord('s'):
			print("Saving Constants for " + str(_img_names[i]))
			# np.savetxt("foo.csv", np.array(csvList), delimiter=",")
			ut.export_list2csv(outDir, outName, csvHeaders, csvList)
		if key == ord('r'):
			print("Data Recording ----- " + str(count_record) + " entries recorded")
			# tmpData = str(_img_names[i] + "," + str(lower_yuv[1]) + "," + str(lower_yuv[2]) + "," + str(lower_yuv[3]) + "," + str(upper_yuv[1]) + "," + str(upper_yuv[2]))
			# tmpData = str(tmpData) + "," + str(upper_yuv[2]) + "," + str(lower_hsv[0]) + "," + str(lower_hsv[1]) + "," + str(lower_hsv[2]) + "," + str(upper_hsv[0])
			# tmpData = str(tmpData) + "," + str(upper_hsv[1]) + "," + str(upper_hsv[2])

			tmpData = [ _img_names[i] , lower_yuv[0], lower_yuv[1], lower_yuv[2], upper_yuv[0], upper_yuv[1], upper_yuv[2], lower_hsv[0], lower_hsv[1], lower_hsv[2], upper_hsv[0], upper_hsv[1], upper_hsv[2]]
			# print tmpData

			print("	YUV Bounds: " + str(lower_yuv) + " --> " + str(upper_yuv))
			print("	HSV Bounds: " + str(lower_hsv) + " --> " + str(upper_hsv))
			csvList.append(tmpData)
			count_record += 1
		if key == ord('p'):
			i = i + 1
			if i >= n:
				i = 0
			print('Next Image ------ ' + str(_img_names[i]))
			tmp = cv2.resize(_imgs[i], (640,480))
		if key == ord('o'):
			i = i - 1
			if i < 0:
				i = n - 1
			print('Previous Image ------ ' + str(_img_names[i]))
			tmp = cv2.resize(_imgs[i], (640,480))

		cv2.imshow("image", tmp)
	cv2.destroyAllWindows()
