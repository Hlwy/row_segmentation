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
	disp_lines = lut.updateLines(img)
	cv2.imshow("Modified Lines Found", disp_lines)
	return disp_lines
	
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
