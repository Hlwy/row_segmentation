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
import numpy as np
from utils import utils as ut
from utils import filter_utils as fut
from utils import seg_utils as sut
from utils import line_utils as lut


def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False

def combine_contours(contours):
	LENGTH = len(contours)
	status = np.zeros((LENGTH,1))
	for i,cnt1 in enumerate(contours):
		x = i
		if i != LENGTH-1:
			for j,cnt2 in enumerate(contours[i+1:]):
				x = x+1
				dist = find_if_close(cnt1,cnt2)
				if dist == True:
					val = min(status[i],status[x])
					status[x] = status[i] = val
				else:
					if status[x]==status[i]:
						status[x] = i+1

	unified = []
	maximum = int(status.max())+1
	for i in xrange(maximum):
		pos = np.where(status==i)[0]
		if pos.size != 0:
			cont = np.vstack(contours[i] for i in pos)
			hull = cv2.convexHull(cont)
			unified.append(hull)

	cv2.drawContours(tmpFil,unified,-1,(255,255,255),2)

def update(img):
	global plt

	# duration = 0
	start = time()

	_img = cv2.resize(img, (640,480))

	horizon_present = sut.is_horizon_present(_img)
	# _img,_ = fut.filter_white(_img)
	filtered_img, _ = update_filter(_img)

	# fut.filter_custom(_img)
	# strips = sut.histogram_strips(filtered_img)

	if horizon_present == True:
		horizon_fit, horizon_inds, horizon_filtered, horizon_display = sut.find_horizon(filtered_img)
		# cv2.imshow("Found Horizon Line", horizon_display)
	else:
		horizon_filtered = filtered_img
		horizon_display = filtered_img

	tmpFil = fut.apply_morph(horizon_filtered, ks=[10,10],flag_open=True)

	h,w,c = tmpFil.shape
	xbuf = 200
	hist = sut.custom_hist(tmpFil,[0,h],[xbuf,w-xbuf],axis=0)
	hist = sut.histogram_sliding_filter(hist)
	xmid = np.argmin(hist[:,0])+xbuf
	# Crop the image into two halves
	beg = xmid; end = w
	img_right = tmpFil[:,beg:end]
	img_left = tmpFil[:,0:beg]

	ret, threshed_img = cv2.threshold(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY),10, 255, cv2.THRESH_BINARY)
	image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# print(hier)
	max_x = 0; min_x = 0; max_y = 0; min_y = 0

	c = max(contours, key = cv2.contourArea)
	hull = cv2.convexHull(c)
	cv2.drawContours(tmpFil, [hull], -1, (0, 0, 255), 1)

	# rows,cols = tmpFil.shape[:2]
	# [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
	# lefty = int((-x*vy/vx) + y)
	# righty = int(((cols-x)*vy/vx)+y)
	# cv2.line(tmpFil,(cols-1,righty),(0,lefty),(0,255,255),2)
    #
	# epsilon = 0.001*cv2.arcLength(c,True)
	# approx = cv2.approxPolyDP(c,epsilon,True)
	# cv2.drawContours(tmpFil, [approx], -1, (255, 0, 255), 1)
    #
	# rect = cv2.minAreaRect(c)
	# box = cv2.boxPoints(rect)
	# box = np.int0(box)
	# cv2.drawContours(tmpFil,[box],0,(255,255,255),2)

	# i = 0
	# for cnt in contours:
	# 	i+=1
	# 	M = cv2.moments(cnt)
	# 	hull = cv2.convexHull(cnt)
	# 	cv2.drawContours(tmpFil, [hull], -1, (0, 0, 255), 1)
	# 	try:
	# 		cX = int(M["m10"] / M["m00"])
	# 		cY = int(M["m01"] / M["m00"])
	# 		cv2.circle(tmpFil, (cX, cY), 2, (255, 255, 255), -1)
	# 		cv2.putText(tmpFil, "center"+str(i), (cX - 20, cY - 20),
	# 			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	# 	except:
	# 		print("ERROR: Moments not found")
	# 		pass



	# for contour, hi in zip(contours, hier):
	# 	(x,y,w,h) = cv2.boundingRect(contour)
	# 	min_x, max_x = min(x, min_x), max(x+w, max_x)
	# 	min_y, max_y = min(y, min_y), max(y+h, max_y)
	# # 	if w > 80 and h > 80:
	# # 		cv2.rectangle(tmpFil, (x,y), (x+w,y+h), (255, 255, 0), 2)
    # #
	# if max_x - min_x > 0 and max_y - min_y > 0:
	# 	cv2.rectangle(tmpFil, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
    #
	# # hull = cv2.convexHull(threshed_img)
	# # cv2.drawContours(tmpFil, [hull], -1, (0, 0, 255), 1)
	# for cnt in contours:
	# 	(x,y,w,h) = cv2.boundingRect(cnt)
	# 	# cv2.rectangle(tmpFil, (x,y), (x+w,y+h), (255, 255, 0), 2)
	# 	# get convex hull
	# 	hull = cv2.convexHull(cnt)
	# 	# draw it in red color
	# 	# cv2.drawContours(tmpFil, [cnt], -1, (0, 255, 0), 1)
	# 	cv2.drawContours(tmpFil, [hull], -1, (0, 0, 255), 1)


	cv2.imshow("morphed",tmpFil)

	# disp_lines, disp_windows = lut.find_line_exp(horizon_filtered)
	disp_lines, _ = lut.ransac_meth2(horizon_filtered,xmid)
	cv2.imshow("Lines Found", disp_lines)
	# cv2.imshow("Windows Used To Find Lines", disp_windows)

	# tmp_disp = np.copy(horizon_filtered)
	# lines = tmp_disp
	# lines = lut.update_zhong(tmp_disp,0, 53.0,0.0,0.0,555.0,577.0)
	# cv2.imshow("Row Lines", lines)

	duration = time() - start
	# print("Processing Duration: " + str(duration))

	return disp_lines


def color_filter(img):
	res,mask = fut.filter_brown(img)
	filtered_img = cv2.bitwise_and(img, img, mask = mask)
	return filtered_img

def update_filter(img, verbose=False):
	global filter_index, mask_flag, use_raw

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
