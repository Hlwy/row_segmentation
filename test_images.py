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


# My pipeline was:
#
# Convert from RGB to grayscale (cvCvtColor)
# Smooth (cvSmooth)
# Threshold (cvThreshold)
# Detect edges (cvCanny)
# Find contours (cvFindContours)
# Approximate contours with linear features (cvApproxPoly)


import cv2
import argparse
from time import time
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from utils import utils as ut
from utils import filter_utils as fut
from utils import seg_utils as sut
from utils import line_utils as lut
from utils import contour_utils as contUtils


def update(img):
	global plt
	# xbuf = 100
	# xL = 0; xR = 0; yminL = 0 ; yminR = 0
	# xLf = 0; xRf = 0; yLf = 0; yRf = 0
	# xR0 = 0; xL0 = 0; yL0 = 0; yR0 = 0
	# ptsL = []; ptsR = []
    #
	# # duration = 0
	# start = time()
    #
	# _img = cv2.resize(img, (640,480))
	# disp_lines = np.copy(_img)
	# h,w,c = _img.shape
    #
	# horizon_present = sut.is_horizon_present(_img)
	# # _img,_ = fut.filter_white(_img)
	# filtered_img, _ = update_filter(_img)
    #
	# if horizon_present == True:
	# 	horizon_fit, horizon_inds, horizon_filtered, horizon_display = sut.find_horizon(filtered_img)
	# 	# cv2.imshow("Found Horizon Line", horizon_display)
	# else:
	# 	horizon_filtered = filtered_img
	# 	horizon_display = filtered_img
    #
	# tmpFil = fut.apply_morph(horizon_filtered, ks=[6,6],flag_open=False)
	# ret, threshed_img = cv2.threshold(cv2.cvtColor(tmpFil, cv2.COLOR_BGR2GRAY),10, 255, cv2.THRESH_BINARY)
	# # tmpDisp1 = np.hstack((_img,filtered_img))
	# # tmpDisp2 = np.hstack((horizon_filtered,tmpFil))
	# # tmpDisp = np.vstack((tmpDisp1,tmpDisp2))
	# # plt.figure(1)
	# # plt.clf()
	# # plt.imshow(tmpDisp)
	# # plt.show()
    #
	# # hist = sut.custom_hist(tmpFil,[0,h],[xbuf,w-xbuf],axis=0, flag_plot=True)
	# hist = sut.custom_hist(threshed_img,[0,h],[xbuf,w-xbuf],axis=0)
	# smoothHist = sut.histogram_sliding_filter(hist, window_size=16)
	# xmid = sut.find_row_ends(smoothHist)
    #
	# # print(xmid[0], xmid[1])
	# # print(xmid.dtype)
    #
	# # Crop the image into two halves
	# try:
	# 	begL = 0
	# 	endL = int(xmid[0])
	# 	begR = int(xmid[1])
	# 	endR = w
	# except:
	# 	begL = 0
	# 	endL = int(xmid)
	# 	begR = endL
	# 	endR = w
    #
	# # img_right = horizon_filtered[:,beg:end]
	# # img_left = horizon_filtered[:,0:beg]
    #
	# img_left = tmpFil[:,begL:endL]
	# img_right = tmpFil[:,begR:endR]
    #
	# ret, threshed_imgL = cv2.threshold(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY),10, 255, cv2.THRESH_BINARY)
	# ret, threshed_imgR = cv2.threshold(cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY),10, 255, cv2.THRESH_BINARY)
    #
	# # cv2.imshow("Image Left",threshed_imgL)
	# # cv2.imshow("Image Right",threshed_imgR)
    #
	# imageL, contoursL, hierL = cv2.findContours(threshed_imgL, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# imageR, contoursR, hierR = cv2.findContours(threshed_imgR, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #
	# cL = max(contoursL, key = cv2.contourArea)
	# cR = max(contoursR, key = cv2.contourArea)
	# cleft = np.array(cL).reshape(-1,2)
	# cright = np.array(cR).reshape(-1,2)
    #
	# careaL = cv2.contourArea(cL)
	# careaR = cv2.contourArea(cR)
    #
	# if careaL < 5000.0:
	# 	# print("Largest Contour to small Looking for lines via RANSAC")
	# 	xmid = w/2
	# 	lineL, lineR, disp_lines,ptsL,ptsR = lut.ransac_meth2(horizon_filtered,xmid,max_trials=20,stop_probability=0.80,display=_img)
	# 	contMins = [0,0]
    #
	# else:
	# 	yminL = 0; yminR = 0
	# 	lineL, lineR, disp_lines,_,_ = lut.ransac_meth2(horizon_filtered,endL,begR,max_trials=30,stop_probability=0.80, display=_img)
    #
	# 	xL0 = np.int(lineL[0][0]); xR0 = np.int(lineR[0][0]) + begR
	# 	yL0 = np.int(lineL[-1][0]); yR0 = np.int(lineR[-1][0])
    #
	# 	xLf = np.int(lineL[0][-1]); xRf = np.int(lineR[0][-1]) + begR
	# 	yLf = np.int(lineL[1][-1]); yRf = np.int(lineR[1][-1])
    #
	# 	if verbose == True:
	# 		print("X Initials: ", xL0,xR0)
	# 		print("Y Initials: ", yL0,yR0)
	# 		print("X Finals: ", xLf,xRf)
	# 		print("Y Finals: ", yLf,yRf)
    #
	# 	# contUtils.show_hulls(img_left,img_right,cL,cR,tmpFil)
    #
	# 	winMins = lut.find_lowest_windows(img_left,img_right,lineL,lineR,flag_beginning=True)
	# 	contMins = sut.find_lowest_contours(cL,cR)
	# 	histMins = sut.find_lowest_histograms(img_left,img_right)
    #
	# 	if contMins[0] > histMins[0]:
	# 		yminL = contMins[0]
	# 	else:
	# 		yminL = histMins[0]
    #
	# 	if winMins[0] > yminL:
	# 		yminL = winMins[0]
    #
	# 	if contMins[1] > histMins[1]:
	# 		yminR = contMins[1]
	# 	else:
	# 		yminR = histMins[1]
    #
	# 	if winMins[1] > yminR:
	# 		yminR = winMins[1]
    #
	# 	if verbose == True:
	# 		print("Chosen Mins: ", yminL,yminR)
    #
	# 	xL = lut.slide_window_right(img_left,[xL0,yminL],threshold=300,size=[30,50])
	# 	xR = lut.slide_window_left(img_right,[xR0,yminR],threshold=300,size=[15,50]) + begR
    #
	# 	winMins2 = lut.find_lowest_windows(img_left,img_right,lineL,lineR,window_size=[40,10],flag_beginning=False)
	# 	yLf = -int(winMins2[0])
	# 	yR0 = -int(winMins2[1])
    #
	# 	ptsL = np.array([ 	[int(xL),int(yminL)],
	# 						[int(xLf),int(-yLf)]
	# 					])
    #
	# 	ptsR = np.array([ 	[int(xR0),int(-yR0)],
	# 						[int(xR),int(yminR)]
	# 					])
    #
	# 	# print(winMins2)
	# 	# cv2.circle(disp_lines,(xL,yminL),2,(255,0,0),-1)
	# 	# cv2.circle(disp_lines,(xR,yminR),2,(0,0,255),-1)
    #
	# 	# cv2.line(disp_lines,(int(xL),int(yminL)),(int(xLf),int(winMins2[0])),(255,255,0))
	# 	# cv2.line(disp_lines,(int(xR0),int(winMins2[1])),(int(xR),int(yminR)),(0,255,255))
    #
	# duration = time() - start
    #
	# ploty = np.linspace(0, w-1, w)
	# left_fit = np.polyfit(ptsL[:,1], ptsL[:,0], 1)
	# right_fit = np.polyfit(ptsR[:,1], ptsR[:,0], 1)
	# plot_leftx = left_fit[0]*ploty + left_fit[1]
	# plot_rightx = right_fit[0]*ploty + right_fit[1]
    #
	# cv2.line(_img,(int(plot_leftx[0]),int(ploty[0])),(int(plot_leftx[-1]),int(ploty[-1])),(255,0,0),thickness=3)
	# cv2.line(_img,(int(plot_rightx[0]),int(ploty[0])),(int(plot_rightx[-1]),int(ploty[-1])),(0,0,255),thickness=3)
	# # cv2.line(_img,(int(xL),int(yminL)),(int(xLf),int(-yLf)),(255,0,0),thickness=3)
	# # cv2.line(_img,(int(xR0),int(-yR0)),(int(xR),int(yminR)),(0,0,255),thickness=3)
    #
	# cv2.imshow("Lines Found: RANSAC Before Modification", disp_lines)
	# cv2.imshow("Lines Found: RANSAC After Modification", _img)
	# # cv2.imshow("morphed",tmpFil)
	# print("Processing Duration: " + str(duration))
	disp_lines = lut.updateLines(img)
	cv2.imshow("Modified Lines Found", disp_lines)
	return disp_lines


def color_filter(img):
	res,mask = fut.filter_brown(img)
	filtered_img = cv2.bitwise_and(img, img, mask = mask)
	return filtered_img

def update_filter(img, filter_index=1,mask_flag=True,use_raw=True, verbose=False):

	# Process the new image
	if filter_index == 0:
		res,mask = fut.filter_green(img)
	elif filter_index == 1:
		res,mask = fut.filter_brown(img)

	if mask_flag == True:
		color_mask = fut.add_green_mask(mask)
		tmp_color_mask = color_mask
	else:
		color_mask = mask
		tmp_color_mask = cv2.cvtColor(color_mask,cv2.COLOR_GRAY2BGR)

	res2 = cv2.bitwise_and(img, img, mask = mask)

	if use_raw == True:
		filtered_img = res2
	else:
		filtered_img = tmp_color_mask

	if verbose == True:
		cv2.imshow("Color Filtered Image", res)
		cv2.imshow("Color Filtered Mask", mask)
		# cv2.imshow("Final Filtered Mask", res2)
		cv2.imshow("Final Filtered Image", res2)

	return filtered_img, mask


#Run Main
if __name__ == "__main__" :

	# Setup commandline argument(s) structures
	ap = argparse.ArgumentParser(description='Image Segmentation')
	ap.add_argument("--pic", "-p", type=str, default='test', metavar='FILE', help="Name of video file to parse")
	# Store parsed arguments into array of variables
	args = vars(ap.parse_args())

	# Extract stored arguments array into individual variables for later usage in script
	_img = args["pic"]
	_imgs, _img_names = ut.get_images_by_dir(_img)
	img = _imgs[0]	# Input video file as OpenCV VideoCapture device

	# create trackbars for color change
	cv2.namedWindow('image')
	plt.ion()
	# plt.figure()

	cur_img = cv2.resize(img, (640,480))
	clone = cv2.resize(img, (640,480))
	display = cv2.resize(img, (640,480))
	filtered_img = display

	count = 1
	duration = 0
	i = 0
	n = len(_imgs)
	filter_index = 1
	max_filters = 2
	mask_flag = True
	use_raw = True
	flag_play = False

	while True:

		key = cv2.waitKey(5) & 0xFF

		if key == ord('p'):
			i = i + 1
			if i >= n:
				i = 0
			print('Next Image...')
			new_img = np.copy(_imgs[i])
			clone = cv2.resize(new_img, (640,480))
			post_img = update(new_img)

		if key == ord('o'):
			i = i - 1
			if i < 0:
				i = n - 1
			print('Previous Image...')
			new_img = np.copy(_imgs[i])
			clone = cv2.resize(new_img, (640,480))
			post_img = update(new_img)

		# Parse the user-input
		if key == ord(' '):
			flag_play = not flag_play
			# filtered_img = update_filter(clone)
			# post_img = update(filtered_img)
		if key == ord('q'):
			break
		if key == ord('j'):
			mask_flag = not mask_flag
			print('Switching Color Mask...')
		if key == ord('r'):
			use_raw = not use_raw
			print('Switching to/from Raw image...')

		# Update new variables
		filter_index = ut.cycle_through_filters(key,filter_index)
		# new_img, new_path, i, flag_new_img = ut.cycle_through_images(key, _imgs, _paths, i)
		cur_img = cv2.resize(clone, (640,480))
		# clone = cv2.resize(new_img, (640,480))

		cv2.imshow("image", cur_img)

		if flag_play == True:
			print("Possible RANSAC Test Section")
			# start = time()
			# filtered_img = update_filter(cur_img)
			# post_img = update(filtered_img)
			# out_img = post_img.astype(np.uint8)
			# cv2.imshow("Filtered Image", filtered_img)
			# plt.show()
			# plt.pause(0.001)


		# cv2.imshow("image", cur_img)
		# plt.imshow(display)
		plt.show()
		plt.pause(0.001)


	cv2.destroyAllWindows()
