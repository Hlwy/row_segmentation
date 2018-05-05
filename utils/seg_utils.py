# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO
from matplotlib import pyplot as plt
import utils as ut
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

	# print("Minimum: " + str(minval))
	tmpVal = int(minval[1])
	idx = int(minval[0])
	return idx

def find_horizon(img, nwindows=8, minpix=300, window_height=20, flag_plot=False):
	tmp = cv2.resize(img, (640,480))
	cropped = np.copy(tmp)
	display = np.copy(tmp)
	h, w, c = tmp.shape

	vhist = vertical_hist(tmp)
	hist = histogram_sliding_filter(vhist)
	minypix = find_horizon_simple(hist[:,1])
	# print("")
	# print(tmp.shape)
	# print("Starting Y-Pixel: " + str(minypix))

	# Set size of the windows
	window_width = np.int(tmp.shape[1]/nwindows)
	# Set the width of the windows +/- margin

	if window_height > minypix:
		window_height = minypix

	# print("Window Size: " + str(window_height) + ", " + str(window_width))

	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = tmp.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Declare initial search coordinates
	x_base = 0# + window_width/2
	y_base = minypix - window_height

	# Check if we need to start our window search on the left or the right side
	win_y_low = y_base - window_height
	win_y_high = y_base + window_height
	win_x_high = x_base + (0+1)*window_width
	win_x_low = x_base + 0*window_width

	good_horizon_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
	(nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

	# print("Start --- # of Good Indices: " + str(len(good_horizon_inds)))
	# Current positions to be updated for each window
	if len(good_horizon_inds) < minpix:
		x_current = w
		flag_left_2_right = False
		# print("Searching for horizon [Right->Left]")
	else:
		x_current = x_base
		flag_left_2_right = True
		# print("Searching for horizon [Left->Right]")

	y_current = y_base

	# Create empty lists to receive left and right lane pixel indices
	horizon_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y
		win_y_low = y_current - window_height
		win_y_high = y_current + window_height
		if flag_left_2_right == True:
			win_x_high = x_current + (window+1)*window_width
			win_x_low = x_current + window*window_width
		else:
			win_x_low = x_current - (window+1)*window_width
			win_x_high = x_current - window*window_width

		# print("	Current Window Center: " + str(x_current) + ", " + str(y_current))
		# print("		Current Window X Limits: " + str(win_x_low) + ", " + str(win_x_high))
		# print("		Current Window Y Limits: " + str(win_y_low) + ", " + str(win_y_high))

		# Draw the windows on the visualization image
		# cv2.rectangle(tmp,(win_x_low,win_y_high),(win_x_high,win_y_low),(255,0,0), 2)

		# Identify the nonzero pixels in x and y within the window
		good_horizon_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
		(nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

		# Append these indices to the lists
		horizon_inds.append(good_horizon_inds)

		# If you found > minpix pixels, recenter next window on their mean position
		# print("Window " + str(window) + "---- # of Good Indices: " + str(len(good_horizon_inds)))
		if len(good_horizon_inds) > minpix:
			x_current = np.int(np.mean(nonzerox[good_horizon_inds]))
			y_current = np.int(np.mean(nonzeroy[good_horizon_inds]))
			# print("New Window Center: " + str(x_current) + ", " + str(y_current))

	# Concatenate the arrays of indices
	horizon_inds = np.concatenate(horizon_inds)

	# Extract left and right line pixel positions
	x = nonzerox[horizon_inds]
	y = nonzeroy[horizon_inds]

	try:
		# Fit a second order polynomial to each
		horizon_fit = np.polyfit(y, x, 1)
		# print("Horizon Slope: " + str(horizon_fit[0]))

		# Generate x and y values for plotting
		ploty = np.linspace(0, tmp.shape[1]-1, tmp.shape[1] )
		plotx = horizon_fit[0]*ploty + horizon_fit[1]
		# plotx = horizon_fit[0]*ploty**2 + horizon_fit[1]*ploty + horizon_fit[2]

		if horizon_fit[0] > 0:
			plotx_offset = horizon_fit[0]*ploty + horizon_fit[1]-200
		else:
			plotx_offset = horizon_fit[0]*ploty + horizon_fit[1]+200


		if flag_plot == True:
			global plt
			fig = plt.figure(3)
			fig.clf()
			fig.imshow(tmp)
			fig.plot(plotx, ploty, color='yellow')
			fig.plot(plotx_offset, ploty, color='red')
			fig.xlim(0, img.shape[1])
			fig.ylim(img.shape[0], 0)
			display = ut.fig2img(fig)
			# plt.show()
			# plt.pause(0.001)
			cv2.imshow("Found Horizon Line",display)
	except:
		print("ERROR: Function 'polyfit' failed!")
		horizon_fit = []
		plotx_offset = [0, 0]
		ploty = [0, 0]
		cropped = tmp
		pass

	pts = np.array([[0,ploty[0]],[plotx_offset[0],ploty[0]],[plotx_offset[-1],ploty[-1]], [cropped.shape[1],0]], np.int32)
	cv2.fillPoly(cropped,[pts],(0,0,0))

	cv2.line(display,(int(plotx_offset[0]),int(ploty[0])),(int(plotx_offset[-1]),int(ploty[-1])),(0,0,255))
	return horizon_fit, horizon_inds, cropped, display


def is_horizon_present(img, nrows=10, verbose=False, flag_plot=False):
	_img = cv2.resize(img, (640,480))

	starting_row = 0

	# rows_right = _img[starting_row:starting_row+nrows,(3*_img.shape[1])//4:]
	# rows_left = _img[starting_row:starting_row+nrows,_img.shape[1]//4:_img.shape[1]//2]

	rows_right = _img[starting_row:starting_row+nrows,(7*_img.shape[1])//8:]
	rows_left = _img[starting_row:starting_row+nrows,_img.shape[1]//8:_img.shape[1]//2]

	hist_right = np.sum(rows_right, axis=0)/nrows
	hist_left = np.sum(rows_left, axis=0)/nrows

	r_avg_right = np.int(np.average(hist_right[0]))
	g_avg_right = np.int(np.average(hist_right[1]))
	b_avg_right = np.int(np.average(hist_right[2]))

	r_avg_left = np.int(np.average(hist_left[0]))
	g_avg_left = np.int(np.average(hist_left[1]))
	b_avg_left = np.int(np.average(hist_left[2]))

	rgb_right = [r_avg_right, g_avg_right, b_avg_right]
	rgb_left = [r_avg_left, g_avg_left, b_avg_left]

	if rgb_left > rgb_right:
		rgb = rgb_left
		if verbose == True:
			print("Left Side of image contains higher RGB values")
	else:
		rgb = rgb_right
		if verbose == True:
			print("Right Side of image contains higher RGB values")

	eps = [200, 200, 200]
	checker = np.greater_equal(rgb,eps)

	if np.all(checker) == True:
		flag = True
		# print("Finding Horizon")
	else:
		flag = False
		# print("No need to find Horizon")
	if verbose == True:
		print("Average RGB Values: " + str(rgb))

	if flag_plot == True:
		global plt
		plt.figure(1)
		plt.clf()
		plt.title('Histogram of the Right 1/4 of the image')
		plt.plot(range(hist_right.shape[0]), hist_right[:,0])
		plt.plot(range(hist_right.shape[0]), hist_right[:,1])
		plt.plot(range(hist_right.shape[0]), hist_right[:,2])
		plt.figure(2)
		plt.clf()
		plt.title('Histogram of the Left 1/4 of the image')
		plt.plot(range(hist_left.shape[0]), hist_left[:,0])
		plt.plot(range(hist_left.shape[0]), hist_left[:,1])
		plt.plot(range(hist_left.shape[0]), hist_left[:,2])

	return flag
