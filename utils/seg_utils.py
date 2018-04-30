# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO
from scipy import stats as sp
from matplotlib import pyplot as plt
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

	# if tmpVal >= -200:
	# 	idx = 0
	# else:
	# 	idx = int(minval[0])

	idx = int(minval[0])

	return idx

def find_horizon(img, nwindows=9, minpix=20, flag_plot=True):
	tmp = cv2.resize(img, (640,480))

	vhist = vertical_hist(img)
	hist = histogram_sliding_filter(vhist)
	minypix = find_horizon_simple(hist[:,1])
	print("")
	print("Starting Y-Pixel: " + str(minypix))
	# Set size of the windows
	window_width = np.int(img.shape[1]/nwindows)
	# Set the width of the windows +/- margin
	window_height = 30
	if window_height > minypix:
		window_height = minypix

	# print("Window Size: " + str(window_height) + ", " + str(window_width))

	flag_left_2_right = True

	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Declare initial search coordinates
	x_base = 0# + window_width/2
	y_base = minypix - window_height
	# Current positions to be updated for each window
	x_current = x_base
	y_current = y_base

	# Create empty lists to receive left and right lane pixel indices
	horizon_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_x_high = x_current + (window+1)*window_width
		win_x_low = x_current + window*window_width
		win_y_low = y_current - window_height
		win_y_high = y_current + window_height

		# print("	Current Window Center: " + str(x_current) + ", " + str(y_current))
		# print("		Current Window X Limits: " + str(win_x_low) + ", " + str(win_x_high))
		# print("		Current Window Y Limits: " + str(win_y_low) + ", " + str(win_y_high))

		# Draw the windows on the visualization image
		cv2.rectangle(tmp,(win_x_low,win_y_high),(win_x_high,win_y_low),(0,255,0), 2)

		# Identify the nonzero pixels in x and y within the window
		good_horizon_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
		(nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

		# Append these indices to the lists
		horizon_inds.append(good_horizon_inds)

		# If you found > minpix pixels, recenter next window on their mean position
		# print("# of Good Indices: " + str(len(good_horizon_inds)))
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

		# Generate x and y values for plotting
		ploty = np.linspace(0, tmp.shape[1]-1, tmp.shape[1] )
		# plotx = horizon_fit[0]*ploty**2 + horizon_fit[1]*ploty + horizon_fit[2]
		plotx = horizon_fit[0]*ploty + horizon_fit[1]

		if flag_plot == True:
			global plt
			# plt.figure()
			plt.clf()
			plt.imshow(tmp)
			plt.plot(plotx, ploty, color='yellow')
			plt.xlim(0, img.shape[1])
			plt.ylim(img.shape[0], 0)
			# plt.show()
			# plt.pause(0.001)
	except:
		print("ERROR: Function 'polyfit' failed!")
		horizon_fit = []
		pass

	return horizon_fit, horizon_inds, tmp


def test_img_row(img, testing_row=0,nrows=1, flag_plot=True):
	_img = cv2.resize(img, (640,480))

	print _img.shape
	rows = _img[testing_row:testing_row+nrows,:]
	# rows = _img[:,testing_row:testing_row+nrows]
	print rows.shape

	hist = np.sum(rows, axis=0)/nrows
	print hist.shape
	# hist = histogram_sliding_filter(hist,2)

	# r_avg = np.int(np.mean(hist[0]))
	# g_avg = np.int(np.mean(hist[1]))
	# b_avg = np.int(np.mean(hist[2]))

	r_avg = np.int(np.average(hist[0]))
	g_avg = np.int(np.average(hist[1]))
	b_avg = np.int(np.average(hist[2]))

	# r_avg = sp.mode(hist[0])
	# g_avg = sp.mode(hist[1])
	# b_avg = sp.mode(hist[2])
	#
	# r_avg = r_avg.mode[0]
	# g_avg = g_avg.mode[0]
	# b_avg = b_avg.mode[0]



	print("Average RGB Values: " + str(r_avg) + ", " + str(g_avg) + ", " + str(b_avg))

	if flag_plot == True:
		global plt
		# plt.figure()
		plt.clf()
		plt.plot(range(hist.shape[0]), hist[:,0])
		plt.plot(range(hist.shape[0]), hist[:,1])
		plt.plot(range(hist.shape[0]), hist[:,2])
