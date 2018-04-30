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
import utils as ut
import filter_utils as fut
import seg_utils as sut

def update(img):

	vhist = sut.vertical_hist(img)
	hist = sut.histogram_sliding_filter(vhist)
	minypix = sut.find_horizon(hist[:,1])

	clipped = sut.crop_below_pixel(img, minypix)
	white = np.ones((clipped.shape[0], clipped.shape[1],3), dtype=np.uint8)*255
	black = np.zeros((minypix, clipped.shape[1],3), dtype=np.uint8)
	mask = np.vstack((black,white))
	mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
	normImg = cv2.bitwise_and(img,img,mask = mask)

	display = ut.update_zhong(normImg,0, 53.0,0.0,0.0,555.0,577.0)

	# cv2.imshow("Clipped Lines", display)
	return display


def update_filter(img):
	global filter_index, mask_flag, use_raw

	# Process the new image
	if filter_index == 0:
		mask = fut.filter_green(img)
	elif filter_index == 1:
		mask = fut.filter_brown(img)

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
	# Store parsed arguments into array of variables
	args = vars(ap.parse_args())

	# Extract stored arguments array into individual variables for later usage in script
	_img = args["pic"]
	_imgs = ut.get_images_by_dir(_img)
	img = _imgs[0]	# Input video file as OpenCV VideoCapture device
	video_path = '/home/hunter/data/vids/1/VID_20170426_145010.mp4'

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
	writer = cv2.VideoWriter("/home/hunter/data/outputs/vids/visual_nav.avi", fourcc, 28.0, (640,480))

	i = 0
	n = len(_imgs)
	filter_index = 1
	max_filters = 2
	mask_flag = True
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
				filtered_img = update_filter(cur_img)
				post_img = update(filtered_img)
				out_img = post_img.astype(np.uint8)

				duration += time() - start
				count += 1
				print("Frame " + str(count))
				print ("FPS: {}".format(count / duration))

				writer.write(out_img)
			cap.release()
			writer.release()
		# cv2.imshow("image", cur_img)
		# plt.imshow(display)
		# plt.show()
		# plt.pause(0.001)

	cv2.destroyAllWindows()
