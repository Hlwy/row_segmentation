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
import numpy as np

from time import time
from sklearn import linear_model, datasets
from matplotlib import pyplot as plt
from utils import utils as ut
from utils import filter_utils as fut
from utils import seg_utils as sut
from utils import line_utils as lut

def update(img):
	global plt

	# duration = 0
	start = time()

	_img = cv2.resize(img, (640,480))
	horizon_present = sut.is_horizon_present(_img)
	# filtered_img = update_filter(_img)
	filtered_img = color_filter(_img)

	if horizon_present == True:
		horizon_fit, horizon_inds, horizon_filtered, horizon_display = sut.find_horizon(filtered_img)
		# cv2.imshow("Found Horizon Line", horizon_display)
	else:
		horizon_filtered = filtered_img
		horizon_display = filtered_img

	# Thresholding
	grey = cv2.cvtColor(horizon_filtered,cv2.COLOR_BGR2GRAY)
	_, mask = cv2.threshold(grey, 150, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	res = cv2.bitwise_and(horizon_filtered, horizon_filtered, mask = mask)
	try:
		disp_lines, _ = lut.ransac_meth2(res)
	except:
		print("ERROR Ransac")
		pass

	duration = time() - start
	# print("Processing Duration: " + str(duration))

	cv2.imshow("Lines Found", disp_lines)
	# cv2.imshow("Windows Used To Find Lines", disp_windows)
	# cv2.imshow("Row Lines", lines)
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
