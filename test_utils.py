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
from matplotlib import pyplot as plt
import numpy as np
import utils as ut
import filter_utils as fut
import seg_utils as sut

def onChange(pos):
	global post_img
	global display
	global plt
	global plot_flag

	# img = _imgs[i]
	# tmp = np.copy(post_img)

	# tmp = cv2.resize(img, (640,480))

	# get current positions of four trackbars
	a = cv2.getTrackbarPos('alpha','image')
	b = cv2.getTrackbarPos('beta','image')
	g = cv2.getTrackbarPos('gamma','image')
	f = cv2.getTrackbarPos('focal','image')
	d = cv2.getTrackbarPos('distance','image')

	a = a * 1.0
	b = b * 1.0
	g = g * 1.0
	f = f * 1.0
	d = d * 1.0

	vhist = sut.vertical_hist(post_img)
	hist = sut.histogram_sliding_filter(vhist)
	minypix = sut.find_horizon(hist[:,1])

	clipped = sut.crop_below_pixel(post_img, minypix)
	white = np.ones((clipped.shape[0], clipped.shape[1],3), dtype=np.uint8)*255
	black = np.zeros((minypix, clipped.shape[1],3), dtype=np.uint8)
	mask = np.vstack((black,white))
	mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
	normImg = cv2.bitwise_and(post_img,post_img,mask = mask)

	# warped, M, Minv = ut.zhong_warp(clipped)
	# unwarped = cv2.warpPerspective(warped, Minv, (tmp.shape[1], tmp.shape[0]))
	# lines = ut.update_zhong(res)
	# display = ut.update_zhong(normImg,minypix, a,b,g,f,d)
	display = ut.update_zhong(normImg,0, a,b,g,f,d)

	# Display images
	# plt.figure()
	# plt.plot(range(hist.shape[0]), hist[:,0])
	# plt.plot(range(hist.shape[0]), hist[:,1])
	# plt.plot(range(hist.shape[0]), hist[:,2])
	plt.imshow(display, cmap="gray")
	cv2.imshow("Clipped Lines", display)


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

	# create trackbars for color change
	cv2.namedWindow('image')
	# create trackbars for color change
	cv2.createTrackbar('alpha','image',45,360,onChange)
	cv2.createTrackbar('beta','image',0,360,onChange)
	cv2.createTrackbar('gamma','image',0,360,onChange)
	cv2.createTrackbar('focal','image',640,1920,onChange)
	cv2.createTrackbar('distance','image',525,1920,onChange)
	cv2.createTrackbar('update','image',0,1,onChange)

	plt.ion()
	plt.figure()

	cur_img = cv2.resize(img, (640,480))
	clone = cv2.resize(img, (640,480))
	display = cv2.resize(img, (640,480))

	i = 0
	n = len(_imgs)
	filter_index = 1
	max_filters = 2
	mask_flag = True
	use_raw = False
	plot_flag = True

	while True:

		key = cv2.waitKey(5) & 0xFF

		# Parse the user-input
		if key == ord(' '):
			# plt.figure(0)
			# plt.plot(range(hist.shape[0]), hist[:,0])
			# plt.plot(range(hist.shape[0]), hist[:,1])
			# plt.plot(range(hist.shape[0]), hist[:,2])
			# plt.show()
			plot_flag = not plot_flag
		if key == ord('q'):
			break
		if key == ord('j'):
			mask_flag = not mask_flag
			print 'Switching Color Mask...'
		if key == ord('r'):
			use_raw = not use_raw
			print 'Switching to/from Raw image...'

		# Update new variables
		filter_index = ut.cycle_through_filters(key,filter_index)
		new_img, i = ut.cycle_through_images(key, _imgs, i)
		cur_img = cv2.resize(new_img, (640,480))
		clone = cv2.resize(new_img, (640,480))

		# cv2.imshow("image", cur_img)

		# Process the new image
		if filter_index == 0:
			res,mask = fut.filter_green(clone)
		elif filter_index == 1:
			res,mask = fut.filter_brown(clone)

		if mask_flag == True:
			color_mask = fut.add_green_mask(mask)
			tmp_color_mask = color_mask
		else:
			color_mask = mask
			tmp_color_mask = cv2.cvtColor(color_mask,cv2.COLOR_GRAY2BGR)

		res2 = cv2.bitwise_and(clone, clone, mask = mask)

		if use_raw == True:
			post_img = res2
		else:
			post_img = tmp_color_mask


		cv2.imshow("image", post_img)
		# plt.imshow(display)
		plt.show()
		plt.pause(0.001)

	cv2.destroyAllWindows()
