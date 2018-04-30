# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO


import numpy as np
import os
import cv2


def crop_bottom_half(image):
	cropped_img = image[image.shape[0]/2:image.shape[0]]
	return cropped_img

def crop_bottom_two_thirds(image):
	cropped_img = image[image.shape[0]/6:image.shape[0]]
	return cropped_img

def crop_below_pixel(image, y_pixel):
	cropped_img = image[y_pixel:image.shape[0]]
	return cropped_img

def horizontal_hist(_img, method=0):
	if method == 1:		# Take a histogram of the bottom half of the image
		hist = np.sum(_img[_img.shape[0]//2:,:], axis=0)
	elif method == 2:	# Take a histogram of the top half of the image
		hist = np.sum(_img[0:_img.shape[0]//2,:], axis=0)
	else:				# Take a histogram of the whole image
		hist = np.sum(_img[:,:], axis=0)
	return hist

def vertical_hist(_img, method=0):
	if method == 1:			# Take a histogram of the left half of the image
		hist = np.sum(_img[_img.shape[1]//2:,:], axis=1)
	elif method == 2:		# Take a histogram of the right half of the image
		hist = np.sum(_img[0:_img.shape[1]//2,:], axis=1)
	else:					# Take a histogram of the whole image
		hist = np.sum(_img[:,:], axis=1)
	return hist


def histogram_sliding_filter(hist, window_size=16):
	n, depth = hist.shape
	avg_hist = np.zeros_like(hist).astype(np.int32)

	sliding_window = np.ones((window_size,))/window_size
	for channel in range(depth):
		tmp_hist = np.convolve(hist[:,channel], sliding_window , mode='same')
		avg_hist[:,channel] = tmp_hist

	return avg_hist



def find_horizon_simple(v_hist,window_size=16):
	minval = [0,0]
	n_slopes = v_hist.shape[0] // window_size

	# print("Slopes: " + str(n_slopes))

	for i in range(n_slopes):
		xold = window_size * (i)
		xnew = window_size * (i+1) - 1

		dx = window_size
		dy = v_hist[xnew] - v_hist[xold]

		m = dy/dx
		tmp = np.array([xnew, m])
		if tmp[1] < minval[1]:
			minval = tmp

	print("Minimum: " + str(minval))

	tmpVal = int(minval[1])

	if tmpVal >= -200:
		idx = 0
	else:
		idx = int(minval[0])
	return idx

def find_horizon(img):
	# Take a histogram of the bottom half of the image
	histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	return left_fit, right_fit, left_lane_inds, right_lane_inds
