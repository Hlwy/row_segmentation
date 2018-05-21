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
import argparse
from time import time
from matplotlib import pyplot as plt
import numpy as np
from utils import utils as ut
from utils import filter_utils as fut
from utils import seg_utils as sut
from utils import line_utils as lut

def update(img):
	global filter_index, mask_flag, use_raw
	_img = cv2.resize(img, (640,480))
	horizon_present = sut.is_horizon_present(_img)
	filtered_img = update_filter(_img)

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
		display,_ = lut.ransac_meth2(res)
		cv2.imshow("Lines Found", display)
	except:
		print("ERROR: Couldn't find lines")
		pass
		display = horizon_filtered
	# cv2.imshow("Clipped Lines", display)
	return display


def update_filter(img):
	global filter_index, mask_flag, use_raw

	# Process the new image
	if filter_index == 0:
		_, mask = fut.filter_green(img)
	elif filter_index == 1:
		_, mask = fut.filter_brown(img)

	if mask_flag == True:
		color_mask = fut.add_green_mask(mask)
		tmp_color_mask = color_mask
	else:
		color_mask = mask
		tmp_color_mask = cv2.cvtColor(color_mask,cv2.COLOR_GRAY2BGR)

	res = cv2.bitwise_and(img, img, mask = mask)

	if use_raw == True:
		filtered_img = res
	else:
		filtered_img = tmp_color_mask

	return filtered_img


#Run Main
if __name__ == "__main__" :

	# Setup commandline argument(s) structures
	ap = argparse.ArgumentParser(description='Image Segmentation')
	ap.add_argument("--pic", "-p", type=str, default='test', metavar='FILE', help="Name of video file to parse")
	ap.add_argument("--out", "-o", type=str, default='test', metavar='FILE', help="Name of video file to parse")
	# Store parsed arguments into array of variables
	args = vars(ap.parse_args())

	# Extract stored arguments array into individual variables for later usage in script
	_img = args["pic"]
	_outputFile = args["out"]
	_imgs, _ = ut.get_images_by_dir(_img)
	img = _imgs[0]	# Input video file as OpenCV VideoCapture device

	# video_path = '/home/hunter/data/vids/early_season/1/VID_20170426_145010.mp4'
	# video_path = '/home/hunter/data/vids/late_season/weeds_in_row/july5-2017/1/2video0.avi'
	# video_path = '/home/hunter/data/training_raw/late_season/weeds_in_row/july5-2017/1/2video0.avi'
	video_path = '/home/hunter/data/training_raw/early_season/5/VID_20170426_144800.mp4'

	# create trackbars for color change
	cv2.namedWindow('image')

	cur_img = cv2.resize(img, (640,480))
	clone = cv2.resize(img, (640,480))
	display = cv2.resize(img, (640,480))

	count = 1
	duration = 0
	cap = cv2.VideoCapture(video_path)
	assert(cap.isOpened())
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	writer = cv2.VideoWriter("/home/hunter/data/outputs/vids/visual_nav_" + str(_outputFile) + ".avi", fourcc, 28.0, (640,480))

	i = 0
	n = len(_imgs)
	filter_index = 1
	max_filters = 2
	mask_flag = False
	use_raw = True
	flag_play = False

	while True:

		key = cv2.waitKey(5) & 0xFF

		# Parse the user-input
		if key == ord(' '):
			flag_play = not flag_play
			# filtered_img = update_filter(clone)
			# post_img = update(filtered_img)
		if key == ord('q'):
			break
		if key == ord('j'):
			mask_flag = not mask_flag
			print 'Switching Color Mask...'
		if key == ord('r'):
			use_raw = not use_raw
			print 'Switching to/from Raw image...'

		# Update new variables
		# filter_index = ut.cycle_through_filters(key,filter_index)
		# new_img, i = ut.cycle_through_images(key, _imgs, i)
		# cur_img = cv2.resize(new_img, (640,480))
		# clone = cv2.resize(new_img, (640,480))

		# cv2.imshow("image", cur_img)

		if flag_play == True:
			while cap.grab():
				start = time()

				_, img = cap.retrieve()
				cur_img = cv2.resize(img, (640,480))
				post_img = lut.updateLines(cur_img)
				out_img = post_img.astype(np.uint8)

				duration += time() - start
				count += 1
				print("Frame " + str(count))
				print ("FPS: {}".format(count / duration))

				writer.write(out_img)
			cap.release()
			writer.release()


	cv2.destroyAllWindows()
