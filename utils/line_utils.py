# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO
import utils as ut
import filter_utils as fut
import seg_utils as sut
import contour_utils as contUtils

from matplotlib import pyplot as plt
from sklearn import linear_model, datasets, neighbors
# from sklearn.neighbors import NearestNeighbors

import numpy as np
import os
import cv2
import math
import random

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_fitted = [np.array([False])]
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def check_detected(self):
        if (self.diffs[0] < 0.01 and self.diffs[1] < 10.0 and self.diffs[2] < 1000.) and len(self.recent_fitted) > 0:
            return True
        else:
            return False


    def update(self,fit):
        if fit is not None:
            if self.best_fit is not None:
                self.diffs = abs(fit - self.best_fit)
                if self.check_detected():
                    self.detected =True
                    if len(self.recent_fitted)>10:
                        self.recent_fitted = self.recent_fitted[1:]
                        self.recent_fitted.append(fit)
                    else:
                        self.recent_fitted.append(fit)
                    self.best_fit = np.average(self.recent_fitted, axis=0)
                    self.current_fit = fit
                else:
                    self.detected = False
            else:
                self.best_fit = fit
                self.current_fit = fit
                self.detected=True
                self.recent_fitted.append(fit)

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
	best_ic = 0
	best_model = None
	random.seed(random_seed)
	for i in xrange(max_iterations):
		s = random.sample(data, int(sample_size))
		m = estimate(s)
		ic = 0
		for j in xrange(len(data)):
			if is_inlier(m, data[j]):
				ic += 1

		# print s
		# print 'estimate:', m,
		# print '# inliers:', ic

		if ic > best_ic:
			best_ic = ic
			best_model = m
			if ic > goal_inliers and stop_at_goal:
				break
	# print 'took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic
	return best_model, best_ic

def augment(xys):
	axy = np.ones((len(xys), 3))
	axy[:, :2] = xys
	return axy

def estimate(xys):
	axy = augment(xys[:2])
	return np.linalg.svd(axy)[-1][-1, :]

def is_inlier(coeffs, xy, threshold):
	return np.abs(coeffs.dot(augment([xy]).T)) < threshold

def test_ransac_line_fit(img):

	n = 100
	max_iterations = 100
	goal_inliers = n * 0.3

	# test data
	xys = np.random.random((n, 2)) * 10
	xys[:50, 1:] = xys[:50, :1]

	pylab.scatter(xys.T[0], xys.T[1])

	# RANSAC
	m, b = lut.run_ransac(xys, estimate, lambda x, y: is_inlier(x, y, 0.01), goal_inliers, max_iterations, 20)
	a, b, c = m
	plt.plot([0, 10], [-c/b, -(c+10*a)/b], color=(0, 1, 0))

def nearestneighborsMeth2(img,midpoint=None,n_neighbors=2, display=None):
	tmp = np.copy(img)

	try:
		d = display.dtype
		display_lines = np.copy(display)
	except:
		display_lines = np.copy(img)

	h,w,c = img.shape

	if midpoint == None:
		beg = w/2; end = w
	else:
		beg = midpoint; end = w

	# Crop the image into two halves
	img_right = img[:,beg:end]
	img_left = img[:,0:beg]

	# cv2.imshow("Cropped Right", img_right)
	# cv2.imshow("Cropped Left", img_left)

	nonzero_left = img_left.nonzero()
	nonzeroy_left = np.array(nonzero_left[0]).reshape(-1,1)
	nonzerox_left = np.array(nonzero_left[1]).reshape(-1,1)

	nonzero_right = img_right.nonzero()
	nonzeroy_right = np.array(nonzero_right[0]).reshape(-1,1)
	nonzerox_right = np.array(nonzero_right[1]).reshape(-1,1)

	# Robustly fit linear model with RANSAC algorithm
	# knnL = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
	# knnR = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')

	knnL = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
	knnR = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')

	# ransacL = linear_model.RANSACRegressor(residual_threshold=None, max_trials=20, stop_probability=0.80)
	# ransacR = linear_model.RANSACRegressor(residual_threshold=None, max_trials=20, stop_probability=0.80)

	knnL.fit(nonzerox_left, -nonzeroy_left)
	knnR.fit(nonzerox_right, -nonzeroy_right)

	# ransacL.fit(nonzerox_left, -nonzeroy_left)
	# ransacR.fit(nonzerox_right, -nonzeroy_right)

	# Predict data of estimated models
	line_xL = np.arange(nonzerox_left.min(), nonzerox_left.max())[:, np.newaxis]
	line_xR = np.arange(nonzerox_right.min(), nonzerox_right.max())[:, np.newaxis]
	line_yL = knnL.predict(line_xL)
	line_yR = knnR.predict(line_xR)

	# line_yL = ransacL.predict(line_xL)
	# line_yR = ransacR.predict(line_xR)

	cv2.line(display_lines,(int(line_xR[0]+beg),int(-line_yR[0])),(int(line_xR[-1]+beg),int(-line_yR[-1])),(0,0,255))
	cv2.line(display_lines,(int(line_xL[0]),int(-line_yL[0])),(int(line_xL[-1]),int(-line_yL[-1])),(255,0,0))

	# print("Estimated coefficients (true, linear regression, RANSAC):")
	# print(ransac.estimator_.coef_)
	# return line_X, line_y_ransac
	return display_lines, line_yL

def ransac_meth2(img,midpointL=None,midpointR=None,max_trials=50,stop_probability=0.80, display=None,y_offset=0):
	tmp = np.copy(img)
	h,w,c = img.shape
	begL = 0; begR = 0
	endL = 0; endR = w

	try:
		d = display.dtype
		display_lines = np.copy(display)
	except:
		display_lines = np.copy(img)

	try:
		# d = midpointL.dtype
		begL = 0; endL = midpointL
	except:
		begL = 0; endL = w/2;

	try:
		d = midpointR.dtype
		begR = midpointR; endR = w
	except:
		begR = endL; endR = w

	# print("Line Finding Limits:" , begL,endL,begR,endR)

	# Crop the image into two halves
	img_left = img[:,begL:endL]
	img_right = img[:,begR:endR]

	# cv2.imshow("Cropped Right", img_right)
	# cv2.imshow("Cropped Left", img_left)

	nonzero_left = img_left.nonzero()
	nonzeroy_left = np.array(nonzero_left[0]).reshape(-1,1)
	nonzerox_left = np.array(nonzero_left[1]).reshape(-1,1)

	nonzero_right = img_right.nonzero()
	nonzeroy_right = np.array(nonzero_right[0]).reshape(-1,1)
	nonzerox_right = np.array(nonzero_right[1]).reshape(-1,1)

	# Robustly fit linear model with RANSAC algorithm
	ransacL = linear_model.RANSACRegressor(residual_threshold=None, max_trials=50, stop_probability=0.90)
	ransacR = linear_model.RANSACRegressor(residual_threshold=None, max_trials=50, stop_probability=0.90)
	ransacL.fit(nonzerox_left, -nonzeroy_left)
	ransacR.fit(nonzerox_right, -nonzeroy_right)
	inlier_maskL = ransacL.inlier_mask_
	inlier_maskR = ransacR.inlier_mask_
	outlier_maskL = np.logical_not(inlier_maskL)
	outlier_maskR = np.logical_not(inlier_maskR)
	# Predict data of estimated models
	line_xL = np.arange(nonzerox_left.min(), nonzerox_left.max())[:, np.newaxis]
	line_xR = np.arange(nonzerox_right.min(), nonzerox_right.max())[:, np.newaxis]
	line_yL = ransacL.predict(line_xL)

	line_yR = ransacR.predict(line_xR)
	line_xR = np.arange(nonzerox_right.min(), nonzerox_right.max())[:, np.newaxis]


	cv2.line(display_lines,(int(line_xR[0] + begR),int(-line_yR[0] + y_offset)),(int(line_xR[-1] + begR),int(-line_yR[-1] + y_offset)),(0,0,255),thickness=3)
	cv2.line(display_lines,(int(line_xL[0]),int(-line_yL[0] + y_offset)),(int(line_xL[-1]),int(-line_yL[-1] + y_offset)),(255,0,0),thickness=3)

	endpointsL = np.array([ [int(line_xL[0]),int(-line_yL[0] + y_offset)],
							[int(line_xL[-1]),int(-line_yL[-1] + y_offset)]
						 ])

	endpointsR = np.array([ [int(line_xR[0] + begR),int(-line_yR[0] + y_offset)],
							[int(line_xR[-1] + begR),int(-line_yR[-1] + y_offset)]
						 ])

	# print("Estimated coefficients (true, linear regression, RANSAC):")
	# print(ransac.estimator_.coef_)
	# return line_X, line_y_ransac
	return [line_xL,line_yL], [line_xR,line_yR],display_lines, endpointsL,endpointsR

def find_lowest_windows(img_left,img_right,line_left,line_right,midpointL=None,midpointR=None, window_size=[30,30], flag_beginning=True,verbose=False):
	lineL = np.array(line_left); lineR = np.array(line_right)

	if flag_beginning == True:
		# Find the value found by sliding a window down starting at the beginning coordinate for the lines
		locL = lineL[:,0]; locR = lineR[:,-1]
		winMinL = slide_window_down(img_left,locL,size=window_size)
		winMinR = slide_window_down(img_right,locR,size=window_size)
		if verbose == True:
			print("Sliding Window Mins @ Beginning: ",winMinL,winMinR)

		# Find the value found by sliding a window down starting at the middle coordinate for the lines
		rL,cL = lineL.shape[:2]
		rR,cR = lineR.shape[:2]
		if verbose == True:
			print("Shape Left", rL, cL)
			print("Shape Right", rR, cR)

		midIdxL = cL // 2
		midIdxR = cR // 2
		if verbose == True:
			print("Left Middle Index", midIdxL)
			print("Right Middle Index", midIdxR)

		locL = lineL[:,midIdxL]; locR = lineR[:,midIdxR]
		winMinMidL = slide_window_down(img_left,locL,size=window_size)
		winMinMidR = slide_window_down(img_right,locR,size=window_size)

		if verbose == True:
			print("Sliding Window Mins @ Middle: ",winMinMidL,winMinMidR)

		if winMinMidL > winMinL:
			winMinL = winMinMidL
		if winMinMidR > winMinR:
			winMinR = winMinMidR
	else:
		locL = lineL[:,-1]; locR = lineR[:, 0]
		winMinL = slide_window_down(img_left,locL,size=window_size)
		winMinR = slide_window_down(img_right,locR,size=window_size)
		if verbose == True:
			print("Locations Used @ End: ",locL,locR)
			print("Sliding Window Mins @ End: ",winMinL,winMinR)

	return [winMinL,winMinR]

def slide_window_down(img,location,size,threshold=400,display=None, verbose=False,flag_show=False):
	good_inds = []
	window = 0
	flag_done = False
	tmp = np.copy(img)
	try:
		d = display.dtype
		display_windows = np.copy(display)
	except:
		display_windows = np.copy(img)

	window_height,window_width = size
	maxheight,maxwidth = tmp.shape[:2]

	x_current = abs(int(location[0])); 	y_current = abs(int(location[1]))

	# Move x-coordinate if we are at the edge so we can capture a bit more information
	if x_current <= window_width:
		x_current = window_width
	elif x_current + window_width > maxwidth:
		x_current = maxwidth - window_width
	elif x_current - window_width < 0:
		x_current = window_width
	if verbose == True:
		print("Starting Location: ", x_current, y_current)

	nonzero = tmp.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	while not flag_done:
		prev_good_inds = good_inds

		if y_current >= maxheight:
			flag_done = True
			if verbose == True:
				print("Exiting: Reached max image height.")
			continue

		# Check for [Y] edge conditions
		if y_current - window_height >= 0:
			win_y_low = y_current - window_height
		else:
			win_y_low = 0

		if y_current + window_height <= maxheight:
			win_y_high = y_current + window_height
		else:
			win_y_high = maxheight

		# Check for [X] edge conditions
		if x_current - window_width >= 0:
			win_x_low = x_current - window_width
		else:
			win_x_low = 0

		if x_current + window_width <= maxwidth:
			win_x_high = x_current + window_width
		else:
			win_x_high = maxwidth

		cv2.circle(display_windows,(x_current,y_current),2,(255,0,0),-1)
		cv2.rectangle(display_windows,(win_x_low,win_y_high),(win_x_high,win_y_low),(255,0,0), 2)

		# Identify the nonzero pixels in x and y within the window
		good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
		(nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

		if verbose == True:
			print("	Current Window [" + str(window) + "] Center: " + str(x_current) + ", " + str(y_current))
			print("		Current Window X Limits: " + str(win_x_low) + ", " + str(win_x_high))
			print("		Current Window Y Limits: " + str(win_y_low) + ", " + str(win_y_high))
			print("		Current Window # Good Pixels: " + str(len(good_inds)))

		if len(good_inds) > threshold:
			y_current = y_current + window_height
		else:
			if verbose == True:
				print("	Not enough good pixels @ " + str(x_current) + ", " + str(y_current))
			try:
				y_mean = np.int(np.mean(nonzeroy[good_inds]))
				y_current = y_mean
			except:
				try:
					y_mean = np.int(np.mean(nonzeroy[prev_good_inds]))
					y_current = y_mean # + window_height
				except:
					y_mean = y_current
					y_current = y_current
			if verbose == True:
				print("Moving Y current to Mean Y Pixels: ", y_mean)
			cv2.circle(display_windows,(x_current,y_current),2,(255,255,0),-1)
			flag_done = True

	if verbose == True:
		print("Last Y found: ", y_current)
	if flag_show == True:
		cv2.imshow("Sliding Window Down",display_windows)
	return y_current


def slide_window_right(img,location,size=[30,30],threshold=400,display=None, verbose=False,flag_show=False):
	good_inds = []
	window = 0
	flag_done = False
	tmp = np.copy(img)
	try:
		d = display.dtype
		display_windows = np.copy(display)
	except:
		display_windows = np.copy(img)

	window_height,window_width = size
	maxheight,maxwidth = tmp.shape[:2]

	x_current = abs(int(location[0])); 	y_current = abs(int(location[1]))

	# Move x-coordinate if we are at the edge so we can capture a bit more information
	if x_current <= window_width:
		x_current = window_width
	if y_current >= maxheight:
		y_current = maxheight - window_height
	if verbose == True:
		print("Starting Location: ", x_current, y_current)

	nonzero = tmp.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	while not flag_done:
		prev_good_inds = good_inds

		if x_current >= maxwidth:
			flag_done = True
			if verbose == True:
				print("Exiting: Reached max image width.")
			continue

		# Check for [Y] edge conditions
		if y_current - window_height >= 0:
			win_y_low = y_current - window_height
		else:
			win_y_low = 0

		if y_current + window_height <= maxheight:
			win_y_high = y_current + window_height
		else:
			win_y_high = maxheight

		# Check for [X] edge conditions
		if x_current - window_width >= 0:
			win_x_low = x_current - window_width
		else:
			win_x_low = 0

		if x_current + window_width <= maxwidth:
			win_x_high = x_current + window_width
		else:
			win_x_high = maxwidth

		cv2.circle(display_windows,(x_current,y_current),2,(255,255,255),-1)
		cv2.rectangle(display_windows,(win_x_low,win_y_high),(win_x_high,win_y_low),(255,0,0), 2)

		# Identify the nonzero pixels in x and y within the window
		good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
		(nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

		if verbose == True:
			print("	Current Window [" + str(window) + "] Center: " + str(x_current) + ", " + str(y_current))
			print("		Current Window X Limits: " + str(win_x_low) + ", " + str(win_x_high))
			print("		Current Window Y Limits: " + str(win_y_low) + ", " + str(win_y_high))
			print("		Current Window # Good Pixels: " + str(len(good_inds)))

		if len(good_inds) > threshold:
			x_current = x_current + window_width
		else:
			if verbose == True:
				print("	Not enough good pixels @ " + str(x_current) + ", " + str(y_current))
			try:
				x_mean = np.int(np.mean(nonzerox[good_inds]))
				x_current = x_mean
			except:
				try:
					x_mean = np.int(np.mean(nonzerox[prev_good_inds]))
					x_current = x_mean
				except:
					x_mean = x_current
					x_current = x_current
			if verbose == True:
				print("Moving X current to Mean X Pixels: ", x_mean)
			cv2.circle(display_windows,(x_current,y_current),2,(255,255,0),-1)
			flag_done = True

	if verbose == True:
		print("Last X found: ", x_current)
	if flag_show == True:
		cv2.imshow("Sliding Window Right",display_windows)
	return x_current


def slide_window_left(img,location,size=[30,30],threshold=400,display=None, verbose=False,flag_show=False):
	good_inds = []
	window = 0
	flag_done = False
	tmp = np.copy(img)
	try:
		d = display.dtype
		display_windows = np.copy(display)
	except:
		display_windows = np.copy(img)

	window_height,window_width = size
	maxheight,maxwidth = tmp.shape[:2]

	x_current = abs(int(location[0])); 	y_current = abs(int(location[1]))

	# Move x-coordinate if we are at the edge so we can capture a bit more information
	if x_current >= maxwidth:
		x_current = maxwidth - window_width
	if y_current >= maxheight:
		y_current = maxheight - window_height
	if verbose == True:
		print("Starting Location: ", x_current, y_current)

	nonzero = tmp.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	while not flag_done:
		prev_good_inds = good_inds

		if x_current <= 0:
			flag_done = True
			if verbose == True:
				print("Exiting: Reached max image width.")
			continue

		# Check for [Y] edge conditions
		if y_current - window_height >= 0:
			win_y_low = y_current - window_height
		else:
			win_y_low = 0

		if y_current + window_height <= maxheight:
			win_y_high = y_current + window_height
		else:
			win_y_high = maxheight

		# Check for [X] edge conditions
		if x_current - window_width >= 0:
			win_x_low = x_current - window_width
		else:
			win_x_low = 0

		if x_current + window_width <= maxwidth:
			win_x_high = x_current + window_width
		else:
			win_x_high = maxwidth

		cv2.circle(display_windows,(x_current,y_current),2,(255,255,255),-1)
		cv2.rectangle(display_windows,(win_x_low,win_y_high),(win_x_high,win_y_low),(0,0,255), 2)

		# Identify the nonzero pixels in x and y within the window
		good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
		(nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

		if verbose == True:
			print("	Current Window [" + str(window) + "] Center: " + str(x_current) + ", " + str(y_current))
			print("		Current Window X Limits: " + str(win_x_low) + ", " + str(win_x_high))
			print("		Current Window Y Limits: " + str(win_y_low) + ", " + str(win_y_high))
			print("		Current Window # Good Pixels: " + str(len(good_inds)))

		if len(good_inds) > threshold:
			x_current = x_current - window_width
		else:
			if verbose == True:
				print("	Not enough good pixels @ " + str(x_current) + ", " + str(y_current))
			try:
				x_mean = np.int(np.mean(nonzerox[good_inds]))
				x_current = x_mean
			except:
				try:
					x_mean = np.int(np.mean(nonzerox[prev_good_inds]))
					x_current = x_mean - window_width
				except:
					x_mean = x_current
					x_current = x_current
			if verbose == True:
				print("Moving X current to Mean X Pixels: ", x_mean)
			cv2.circle(display_windows,(x_current,y_current),2,(0,255,255),-1)
			flag_done = True

	if verbose == True:
		print("Last X found: ", x_current)
	if flag_show == True:
		cv2.imshow("Sliding Window Left",display_windows)
	return x_current

def find_line_exp(img, margins=[150,75], nwindows=20, minpix=300, flag_plot=False, flag_manual=False, flag_plot_hists=False):
	tmp = cv2.resize(img, (640,480))
	display_windows = np.copy(tmp)
	display_lines = np.copy(tmp)
	h, w, c = tmp.shape

	h_final = 15
	w_final = 30

	xbuf=margins[0]
	ybuf=margins[1]

	# # Look at the first 1/8 of both the image's sides
	# rows_right = tmp[:,(7*w)//8:]
	# rows_left = tmp[:,0:w//8]
	# Look at the first 1/4 of both the image's sides
	rows_right = tmp[:,(3*w)//4:]
	rows_left = tmp[:,0:w//4]

	hist_right = np.sum(rows_right, axis=1)
	hist_left = np.sum(rows_left, axis=1)

	if flag_plot_hists == True:
		plt.figure(5)
		plt.clf()
		plt.subplot(1,2,1)
		plt.title("Histogram: Left Side of the Image")
		plt.plot(range(hist_left.shape[0]), hist_left[:,0])
		plt.plot(range(hist_left.shape[0]), hist_left[:,1])
		plt.plot(range(hist_left.shape[0]), hist_left[:,2])
		plt.subplot(1,2,2)
		plt.title("Histogram: Right Side of the Image")
		plt.plot(range(hist_right.shape[0]), hist_right[:,0])
		plt.plot(range(hist_right.shape[0]), hist_right[:,1])
		plt.plot(range(hist_right.shape[0]), hist_right[:,2])

	eps=300
	flag_found = False
	for i in range(h-1,-1,-1):
		if flag_found == False:
			tmpPixL = hist_left[i,1]
			if tmpPixL > eps:
				lefty_base = i
				flag_found = True
				# print("Left Line: Starting Pixel Found @ i = " + str(i))
				# print("Left Line: Histogram Value @ " + str(lefty_base) + ": " + str(tmpPixL))

	flag_found = False
	for i in range(h-1,-1,-1):
		if flag_found == False:
			tmpPixR = hist_right[i,1]
			if tmpPixR > eps:
				righty_base = i
				flag_found = True
				# print("Right Line: Starting Pixel Found @ i = " + str(i))
				# print("Right Line: Histogram Value @ " + str(righty_base) + ": " + str(tmpPixR))

	lefty_base = lefty_base - 20
	righty_base = righty_base - 20
	# print("Left Base Y Pixel Found @ " + str(lefty_base))
	# print("Right Base Y Pixel Found @ " + str(righty_base))

	# Set height of windows
	height_base = ybuf
	width_base = xbuf

	window_height_l = height_base
	window_width_l = width_base
	window_height_r = height_base
	window_width_r = width_base
	# print("Window Size: " + str(window_height) + ", " + str(window_width))

	leftx_base = 0 #+ window_width
	rightx_base = w #- window_width

	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = tmp.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	lefty_current = lefty_base
	rightx_current = rightx_base
	righty_current = righty_base

	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	factor = 1
	# Step through the windows one by one
	for window in range(nwindows):

		# Left Side:
		#			Check for [Y] edge conditions
		if lefty_current - window_height_l >= 0:
			win_yleft_low = lefty_current - window_height_l
		else:
			win_yleft_low = 0

		if lefty_current + window_height_l <= h:
			win_yleft_high = lefty_current + window_height_l
		else:
			win_yleft_high = h

		#			Check for [X] edge conditions
		if leftx_current - window_width_l >= 0:
			win_xleft_low = leftx_current - window_width_l
		else:
			win_xleft_low = 0

		if leftx_current + window_width_l <= w:
			win_xleft_high = leftx_current + window_width_l
		else:
			win_xleft_high = w

		# Right Side:
		#			Check for [Y] edge conditions
		if righty_current - window_height_r >= 0:
			win_yright_low = righty_current - window_height_r
		else:
			win_yright_low = 0

		if righty_current + window_height_r <= h:
			win_yright_high = righty_current + window_height_r
		else:
			win_yright_high = h

		#			Check for [X] edge conditions
		if rightx_current - window_width_r >= 0:
			win_xright_low = rightx_current - window_width_r
		else:
			win_xright_low = 0

		if rightx_current + window_width_r <= w:
			win_xright_high = rightx_current + window_width_r
		else:
			win_xright_high = w

		cv2.circle(display_windows,(leftx_current,lefty_current),2,(0,0,255),-1)
		cv2.circle(display_windows,(rightx_current,righty_current),2,(255,0,0),-1)
		cv2.rectangle(display_windows,(win_xleft_low,win_yleft_high),(win_xleft_high,win_yleft_low),(0,0,255), 2)
		cv2.rectangle(display_windows,(win_xright_low,win_yright_high),(win_xright_high,win_yright_low),(255,0,0), 2)

		# print("	Current Window Center: " + str(x_current) + ", " + str(y_current))
		# print("		Current Window X Limits: " + str(win_x_low) + ", " + str(win_x_high))
		# print("		Current Window Y Limits: " + str(win_y_low) + ", " + str(win_y_high))

		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_yleft_low) & (nonzeroy < win_yleft_high) &
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_yright_low) & (nonzeroy < win_yright_high) &
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		# window_height = int(h_final + (height_base* factor * math.exp(-(window+1)/1.0)))
		# window_width = int(w_final + (width_base* factor * math.exp(-(window+1)/2.5)))
		# print("Next Window Size: " + str(window_height) + ", " + str(window_width))

		dx = 1
		dy = 2

		# If you found > minpix pixels, recenter next window on their mean position
		# print("Window " + str(window) + "---- # of Good Indices [left,right]: " + str(len(good_left_inds)) + ", " + str(len(good_right_inds)))
		if len(good_left_inds) > minpix:
			window_height_l = int(h_final + (height_base* factor * math.exp(-(window+1)/1.0)))
			window_width_l = int(w_final + (width_base* factor * math.exp(-(window+1)/1.0)))

			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			lefty_current = np.int(np.mean(nonzeroy[good_left_inds])) - window_height_l
			# lefty_current = lefty_current - window_height
			# print("	New Left Window Center: " + str(leftx_current) + ", " + str(lefty_current))
		else:
			window_height_l = int(h_final + (height_base* factor * math.exp(-(window+1)/1.5)))
			window_width_l = int(w_final + (width_base* factor * math.exp(-(window+1)/2.5)))

			try:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds])) + dx*window_width_l
			except:
				leftx_current = leftx_current + dx*window_width_l
			lefty_current = lefty_current - dy*window_height_l
			# print("[Left]---Not enough good pixels moving window center up: " + str(leftx_current) + ", " + str(lefty_current))

		if len(good_right_inds) > minpix:
			window_height_r = int(h_final + (height_base* factor * math.exp(-(window+1)/1.0)))
			window_width_r = int(w_final + (width_base* factor * math.exp(-(window+1)/1.0)))

			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
			righty_current = np.int(np.mean(nonzeroy[good_right_inds])) - window_height_r
			# righty_current = righty_current - window_height
			# print("	New Right Window Center: " + str(rightx_current) + ", " + str(righty_current))
		else:
			window_height_r = int(h_final + (height_base* factor * math.exp(-(window+1)/1.5)))
			window_width_r = int(w_final + (width_base* factor * math.exp(-(window+1)/2.5)))

			try:
				rightx_current = np.int(np.mean(nonzerox[good_right_inds])) - dx*window_width_r
			except:
				rightx_current = rightx_current - dx*window_width_r
			righty_current = righty_current - dy*window_height_r
			# print("[Right]---Not enough good pixels moving window center up: " + str(rightx_current) + ", " + str(righty_current))

		if flag_manual == True:
			cv2.imshow("Lines Found", display_windows)
			while True:
				key = cv2.waitKey(5) & 0xFF
				if key == ord(' '):
					break
				if key == ord('q'):
					return -1
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	ploty = np.linspace(0, w-1, w )

	# Try fitting polynomial to left side
	try:
		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 1)
		# Generate x and y values for plotting
		plot_leftx = left_fit[0]*ploty + left_fit[1]
	except:
		print("ERROR: Function 'polyfit' failed for LEFT SIDE!")
		left_fit = [0, 0]
		plot_leftx = [0, 0]
		pass

	# Try fitting polynomial to Right side
	try:
		# Fit a second order polynomial to each
		right_fit = np.polyfit(righty, rightx, 1)
		# Generate x and y values for plotting
		plot_rightx = right_fit[0]*ploty + right_fit[1]
	except:
		print("ERROR: Function 'polyfit' failed for RIGHT SIDE!")
		right_fit = []
		plot_rightx = [0, 0]
		pass

	cv2.line(display_lines,(int(plot_leftx[0]),int(ploty[0])),(int(plot_leftx[-1]),int(ploty[-1])),(0,0,255))
	cv2.line(display_lines,(int(plot_rightx[0]),int(ploty[0])),(int(plot_rightx[-1]),int(ploty[-1])),(255,0,0))

	if flag_plot == True:
		plt.figure(4)
		plt.clf()
		plt.imshow(display_lines)
		plt.plot(plot_leftx, ploty, color='yellow')
		plt.plot(plot_rightx, ploty, color='yellow')
		plt.xlim(0, w)
		plt.ylim(h, 0)


	return display_lines, display_windows #,left_fit, right_fit, left_lane_inds, right_lane_inds


def find_line_test(img, horizon_y, margins=[150,75]):
	tmp = cv2.resize(img, (640,480))
	display_windows = np.copy(tmp)
	display_lines = np.copy(tmp)
	h, w, c = tmp.shape

	h_final = 15
	w_final = 30

	xbuf=margins[0]
	ybuf=margins[1]

	# # Look at the first 1/8 of both the image's sides
	# rows_right = tmp[:,(7*w)//8:]
	# rows_left = tmp[:,0:w//8]
	# Look at the first 1/4 of both the image's sides
	rows_right = tmp[:,(3*w)//4:]
	rows_left = tmp[:,0:w//4]

	hist_right = np.sum(rows_right, axis=1)
	hist_left = np.sum(rows_left, axis=1)

	if flag_plot_hists == True:
		plt.figure(5)
		plt.clf()
		plt.subplot(1,2,1)
		plt.title("Histogram: Left Side of the Image")
		plt.plot(range(hist_left.shape[0]), hist_left[:,0])
		plt.plot(range(hist_left.shape[0]), hist_left[:,1])
		plt.plot(range(hist_left.shape[0]), hist_left[:,2])
		plt.subplot(1,2,2)
		plt.title("Histogram: Right Side of the Image")
		plt.plot(range(hist_right.shape[0]), hist_right[:,0])
		plt.plot(range(hist_right.shape[0]), hist_right[:,1])
		plt.plot(range(hist_right.shape[0]), hist_right[:,2])

	eps=300
	flag_found = False
	lefty_base = 0
	righty_base = 0
	for i in range(h-1,-1,-1):
		if flag_found == False:
			tmpPixL = hist_left[i,1]
			if tmpPixL > eps:
				lefty_base = i
				flag_found = True
				# print("Left Line: Starting Pixel Found @ i = " + str(i))
				# print("Left Line: Histogram Value @ " + str(lefty_base) + ": " + str(tmpPixL))

	flag_found = False
	for i in range(h-1,-1,-1):
		if flag_found == False:
			tmpPixR = hist_right[i,1]
			if tmpPixR > eps:
				righty_base = i
				flag_found = True
				# print("Right Line: Starting Pixel Found @ i = " + str(i))
				# print("Right Line: Histogram Value @ " + str(righty_base) + ": " + str(tmpPixR))

	lefty_base = lefty_base - 20
	righty_base = righty_base - 20
	# print("Left Base Y Pixel Found @ " + str(lefty_base))
	# print("Right Base Y Pixel Found @ " + str(righty_base))

	# Set height of windows
	height_base = ybuf
	width_base = xbuf
	# print("Window Size: " + str(height_base) + ", " + str(width_base))

	leftx_base = 0 #+ window_width
	rightx_base = w #- window_width

	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = tmp.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	lefty_current = lefty_base
	rightx_current = rightx_base
	righty_current = righty_base


	if flag_plot == True:
		plt.figure(4)
		plt.clf()
		plt.imshow(display_lines)
		plt.plot(plot_leftx, ploty, color='yellow')
		plt.plot(plot_rightx, ploty, color='yellow')
		plt.xlim(0, w)
		plt.ylim(h, 0)


	return display_lines, display_windows


def find_line_by_previous(binary_warped,left_fit,right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
    left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit, left_lane_inds, right_lane_inds

def fitLine(points, row, col):
	[vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

	left = Point(0, int(-x*vy/vx + y))
	right = Point(col-1, int((col-1-x)*vy/vx + y))
	top = Point(int(-y*vx/vy + x), 0)
	bot = Point(int((row-1-y)*vx/vy + x), row-1)

	points = [left, right, top, bot]
	points.sort(key = lambda p : (p.x - x)**2 + (p.y - y)**2)

	return points[0], points[1], [vx, vy, x, y]


def update_zhong(_img,y_intercept, _alpha=45.0, _beta=0.0,_gamma=0.0, _focal=500.0, _dist=500.0):
	img = np.copy(_img)

	tmp, M, Minv = warp_perspective_angles(img, _alpha,_beta,_gamma, _focal,_dist)

	# tmp = img

	[row, col, depth] = tmp.shape
	tmp = tmp.astype(np.uint8)
	# print tmp.dtype
	green = (2 * tmp[:,:,1] - tmp[:,:,0] - tmp[:,:,2])
	green = np.clip(green, 0, 255)
	green = green.astype(np.uint8)
	# print green.dtype

	_, thresh = cv2.threshold(green, 20, 255, cv2.THRESH_BINARY)
	# print thresh.dtype

	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)) # Original
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 3)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations = 1)

	# cv2.imshow('',opening)
	# tmp = tmp.astype(np.uint8)
	# print opening.dtype

	left = np.zeros(opening.shape, opening.dtype)
	right = np.zeros(opening.shape, opening.dtype)
	left[:,:col/2] = opening[:,:col/2].copy()
	right[:,col/2:] = opening[:,col/2:].copy()
	# print right

	points = cv2.findNonZero(left)
	if points is None:
		print("No Points Found [left]")
		return -1
	p1, p2, line_left = fitLine(points, row, col)

	cv2.line(tmp, (p1.x, p1.y+y_intercept), (p2.x, p2.y+y_intercept), (0,0,255), 2)

	points = cv2.findNonZero(right)
	if points is None:
		print("No Points Found [right]")
		return -1
	p1, p2, line_right = fitLine(points, row, col)

	cv2.line(tmp, (p1.x, p1.y+y_intercept), (p2.x, p2.y+y_intercept), (0,0,255), 2)

	mid = np.zeros((row, 1, 2), points.dtype)
	for i in xrange(row):
		x1 = line_left[0] / line_left[1] * (i - line_left[3]) + line_left[2]
		x2 = line_right[0] / line_right[1] * (i - line_right[3]) + line_right[2]
		mid[i] = [[(x1 + x2)/2, i]]

	p1, p2, [vx,vy,x,y] = fitLine(mid, row, col)

	r = np.arctan(vy/vx) * 180 / np.pi
	r = np.sign(r) * (90 - abs(r))
	t = (vy/vx) * (row/2 - y) + x - col/2

	cv2.line(tmp, (p1.x, p1.y), (p2.x, p2.y), (0, 255, 0), 2)
	cv2.line(tmp, (col/2,0), (col/2,row-1), (255, 0, 0), 2)

	# cv2.putText(tmp, "Rotation: " + str(r), (10, 30),
	# 			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
	# cv2.putText(tmp, "Translation: " + str(t), (10, 60),
	# 			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
	# cv2.imshow('show', tmp)
	# print tmp.shape
	return tmp

def warp_perspective_angles(_img, _alpha=45.0, _beta=0.0,_gamma=0.0, _focal=500.0, _dist=500.0):
	alpha = _alpha * np.pi / 180
	beta = _beta * np.pi / 180
	gamma = _gamma * np.pi / 180

	focal = float(_focal)
	dist = float(_dist)

	[row, col, depth] = _img.shape

	# Projection Matrix 2D -> 3D
	A = np.array([[1,0,-col/2],[0,1,-row/2],[0,0,0],[0,0,1]])

	# Rotation Matrices: Rx, Ry, Rz
	Rx = np.array([
			[1,0,0,0],
			[0,np.cos(alpha),-np.sin(alpha),0],
			[0,np.sin(alpha),np.cos(alpha),0],
			[0,0,0,1]
		])

	Ry = np.array([
			[np.cos(beta), 0, -np.sin(beta), 0],
			[0, 1, 0, 0],
			[np.sin(beta), 0, np.cos(beta), 0],
			[0, 0, 0, 1]
		])

	Rz = np.array([
			[np.cos(gamma), -np.sin(gamma), 0, 0],
			[np.sin(gamma), np.cos(gamma), 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]
		])

	# 3D Rotation Matrix
	# R = Rx
	R = np.matmul(Rx, np.matmul(Ry,Rz))

	# Translation Matrix
	T = np.array([
			[1,0,0,0],
			[0,1,0,0],
			[0,0,1,dist],
			[0,0,0,1]
		])

	# Intrinsic Matrix
	K = np.array([
			[focal,0,col/2,0],
			[0,focal,row/2,0],
			[0,0,1,0]
		])

	# Perspective Transform Matrix
	M = np.matmul(K, np.matmul(T, np.matmul(R, A)))

	try:
		Minv = np.linalg.inv(M)
	except np.linalg.LinAlgError:
		Minv = None
		print('Un-Invertible Perspective Transform')
		pass # Not invertible. Skip this one.

	warped = cv2.warpPerspective(_img, M, (col, row))
	warped = warped.astype(np.float32)
	return warped, M, Minv

def warp_perspective_points(_img):

	h, w, channels = _img.shape

	_src = [(78, 324), (216, 131), (553, 126), (635, 343)]
	_dst = [(40, h), (40, 0), (w-40, h), (w-40, 0)]

	# _src = [(28, 175), (117, 127), (515, 117), (608, 189)]
	# _dst = [(0, 480), (0, 0), (640, 480), (543, 0)]

	# _src = [(78, 324), (216, 131), (553, 126), (635, 343)]
	# _dst = [(40, 480), (40, 0), (600, 480), (600, 0)]

	# _src = [(74, 360), (196, 211), (434, 212), (633, 386)]
	# _dst = [(40, 480), (40, 0), (600, 480), (600, 0)]

	src = np.float32([_src])
	dst = np.float32([_dst])

	M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
	Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

	warped = cv2.warpPerspective(_img, M, (w, h)) # Image warping

	return warped, M, Minv

def calculate_curv_and_pos(img,left_fit, right_fit):
	# Define y-value where we want radius of curvature
	ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
	leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	y_eval = np.max(ploty)
	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	curvature = ((left_curverad + right_curverad) / 2)
	#print(curvature)
	lane_width = np.absolute(leftx[719] - rightx[719])
	lane_xm_per_pix = 3.7 / lane_width
	veh_pos = (((leftx[719] + rightx[719]) * lane_xm_per_pix) / 2.)
	cen_pos = ((binary_warped.shape[1] * lane_xm_per_pix) / 2.)
	distance_from_center = cen_pos - veh_pos

	# angle = int(math.atan((y1-y2)/(x2-x1))*180/math.pi) # Random forum code snippet

	return curvature,distance_from_center


def updateLines(img, xbuf=100,verbose=False,timer=False,show_images=True,crop_point=100):
	xL = 0; xR = 0; yminL = 0 ; yminR = 0
	xLf = 0; xRf = 0; yLf = 0; yRf = 0
	xR0 = 0; xL0 = 0; yL0 = 0; yR0 = 0
	tmp = []; filtered_img_crop = []; ptsL = []; ptsR = []
	duration = 0; early_season = False

	if timer:
		start = time()

	_img = cv2.resize(img, (640,480))
	clone = np.copy(_img)

	tmp = sut.crop_below_pixel(_img,crop_point)
	hmax,wmax,cmax = _img.shape

	disp_lines = np.copy(tmp)
	h,w,c = tmp.shape

	horizon_present = sut.is_horizon_present(tmp)
	filtered_img, _ = fut.update_filter(clone)

	if crop_point > 0:
		filtered_img_crop = sut.crop_below_pixel(filtered_img,crop_point)
	else:
		filtered_img_crop = np.copy(filtered_img)

	if horizon_present == True:
		horizon_fit, horizon_inds, horizon_filtered, horizon_display = sut.find_horizon(filtered_img)
		# cv2.imshow("Found Horizon Line", horizon_display)
	else:
		horizon_filtered = filtered_img
		horizon_display = filtered_img

	if crop_point > 0:
		horizon_filtered_crop = sut.crop_below_pixel(horizon_filtered,crop_point)

	tmpFil = fut.apply_morph(horizon_filtered, ks=[5,5],flag_open=False)
	tmpFil_crop = fut.apply_morph(horizon_filtered_crop, ks=[5,5],flag_open=False)
	ret, threshed_img = cv2.threshold(cv2.cvtColor(tmpFil, cv2.COLOR_BGR2GRAY),2, 255, cv2.THRESH_BINARY)

	# tmpDisp1 = np.hstack((_img,filtered_img))
	# tmpDisp2 = np.hstack((horizon_filtered,tmpFil))
	# tmpDisp = np.vstack((tmpDisp1,tmpDisp2))
	# plt.figure(1)
	# plt.clf()
	# plt.imshow(tmpDisp)
	# plt.show()

	hist = sut.custom_hist(threshed_img,[0,h],[xbuf,w-xbuf],axis=0)
	smoothHist = sut.histogram_sliding_filter(hist, window_size=16)
	xmid = sut.find_row_ends(smoothHist)

	# Crop the image into two halves
	try:
		begL = 0
		endL = int(xmid[0])
		begR = int(xmid[1])
		endR = w
	except:
		begL = 0
		endL = int(xmid)
		begR = endL
		endR = w

	img_left = tmpFil[:,begL:endL]
	img_right = tmpFil[:,begR:endR]
	img_left_crop = tmpFil_crop[:,begL:endL]
	img_right_crop = tmpFil_crop[:,begR:endR]

	ret, threshed_imgL = cv2.threshold(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY),10, 255, cv2.THRESH_BINARY)
	ret, threshed_imgR = cv2.threshold(cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY),10, 255, cv2.THRESH_BINARY)

	# cv2.imshow("Image Left",threshed_imgL)
	# cv2.imshow("Image Right",threshed_imgR)

	imageL, contoursL, hierL = cv2.findContours(threshed_imgL, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	imageR, contoursR, hierR = cv2.findContours(threshed_imgR, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	try:
		cL = max(contoursL, key = cv2.contourArea)
		cR = max(contoursR, key = cv2.contourArea)
		cleft = np.array(cL).reshape(-1,2)
		cright = np.array(cR).reshape(-1,2)

		careaL = cv2.contourArea(cL)
		careaR = cv2.contourArea(cR)

	except:
		print("ERROR: function \'updateLines\' could not find contours ")
		return clone

	if careaL < 5000.0:
		# print("Largest Contour to small Looking for lines via RANSAC")
		xmid = w/2
		contMins = [0,0]
		try:
			lineL, lineR, disp_lines,ptsL,ptsR = ransac_meth2(horizon_filtered_crop,xmid,max_trials=20,stop_probability=0.80,display=_img,y_offset=crop_point)
			early_season = True
		except:
			print("ERROR: function \'updateLines\' could not find lines for small contour section")
			return clone
	else:
		yminL = 0; yminR = 0
		try:
			lineL, lineR, disp_lines,_,_ = ransac_meth2(horizon_filtered,endL,begR,max_trials=50,stop_probability=0.80, display=_img,y_offset=0)
			early_season = False

			xL0 = np.int(lineL[0][0]); xR0 = np.int(lineR[0][0]) + begR
			yL0 = np.int(lineL[-1][0]);	yR0 = np.int(lineR[-1][0])
			# yL0 = np.int(lineL[-1][0]) - int(crop_point)
			# yR0 = np.int(lineR[-1][0]) - int(crop_point)

			xLf = np.int(lineL[0][-1]); xRf = np.int(lineR[0][-1]) + begR
			yLf = np.int(lineL[1][-1]);	yRf = np.int(lineR[1][-1])
			# yLf = np.int(lineL[1][-1]) - int(crop_point)
			# yRf = np.int(lineR[1][-1]) - int(crop_point)

			if verbose == True:
				print("X Initials: ", xL0,xR0)
				print("Y Initials: ", yL0,yR0)
				print("X Finals: ", xLf,xRf)
				print("Y Finals: ", yLf,yRf)

			# contUtils.show_hulls(img_left,img_right,cL,cR,tmpFil)

			winMins = find_lowest_windows(img_left,img_right,lineL,lineR,flag_beginning=True)
			contMins = sut.find_lowest_contours(cL,cR)
			histMins = sut.find_lowest_histograms(img_left,img_right)

			if contMins[0] > histMins[0]:
				yminL = contMins[0]
			else:
				yminL = histMins[0]

			if winMins[0] > yminL:
				yminL = winMins[0]

			if contMins[1] > histMins[1]:
				yminR = contMins[1]
			else:
				yminR = histMins[1]

			if winMins[1] > yminR:
				yminR = winMins[1]

			if verbose == True:
				print("Chosen Mins: ", yminL,yminR)

			xL = slide_window_right(img_left,[xL0,yminL],threshold=300,size=[30,50])
			xR = slide_window_left(img_right,[xR0,yminR],threshold=300,size=[15,50]) + begR

			winMins2 = find_lowest_windows(img_left,img_right,lineL,lineR,window_size=[40,10],flag_beginning=False)
			yLf = -int(winMins2[0])
			yR0 = -int(winMins2[1])

			ptsL = np.array([ 	[int(xL),int(yminL)],
								[int(xLf),int(-yLf)]
							])

			ptsR = np.array([ 	[int(xR0),int(-yR0)],
								[int(xR),int(yminR)]
							])
		except:
			print("ERROR: function \'updateLines\' could not find lines for large contour section")
			return clone

	if timer:
		duration = time() - start
		print("Processing Duration: " + str(duration))

	ploty = np.linspace(0, w-1, w)
	left_fit = np.polyfit(ptsL[:,1], ptsL[:,0], 1)
	right_fit = np.polyfit(ptsR[:,1], ptsR[:,0], 1)
	plot_leftx = left_fit[0]*ploty + left_fit[1]
	plot_rightx = right_fit[0]*ploty + right_fit[1]

	tmp = clone
	# if early_season == False:
	# 	tmp = clone

	cv2.line(tmp,(int(plot_leftx[0]),int(ploty[0])),(int(plot_leftx[-1]),int(ploty[-1])),(255,0,0),thickness=3)
	cv2.line(tmp,(int(plot_rightx[0]),int(ploty[0])),(int(plot_rightx[-1]),int(ploty[-1])),(0,0,255),thickness=3)
	# cv2.line(_img,(int(xL),int(yminL)),(int(xLf),int(-yLf)),(255,0,0),thickness=3)
	# cv2.line(_img,(int(xR0),int(-yR0)),(int(xR),int(yminR)),(0,0,255),thickness=3)

	line_left = np.array([plot_leftx,ploty])
	line_right = np.array([plot_rightx,ploty])

	# print line_left[:,0]
	# print line_left[:,-1]
	# print line_right[:,0]
	# print line_right[:,-1]

	# print hmax
	line_mid_coef, line_mid, midLinePts = find_middle_line(line_left,line_right,hmax)
	line_mid_ref_coef, intersect, refPts = find_middle_reference_line(line_left,line_right,hmax)

	theta_lines, x_intercept = find_line_angle(midLinePts,refPts)
	print("	Navigation Line [Angle(deg), X-Intercept(px)]: " + str([theta_lines*180/math.pi, x_intercept]))
	plot_line_ref = line_mid_ref_coef[0]*ploty + line_mid_ref_coef[1]
	line_ref = np.array([plot_line_ref,ploty])
	cv2.line(tmp,(int(line_mid[0,0]),int(line_mid[1,0])),(int(line_mid[0,-1]),int(line_mid[1,-1])),(0,255,255),thickness=3)
	cv2.line(tmp,(int(line_ref[0,0]),int(line_ref[1,0])),(int(line_ref[0,-1]),int(line_ref[1,-1])),(35,128,58),thickness=3)
	cv2.circle(tmp,(int(intersect[0]),int(intersect[1])),5,(255,255,0),-1)
	cv2.circle(tmp,(int(line_mid[0,-1]),int(hmax)),5,(255,0,255),-1)
	cv2.circle(tmp,(int(wmax/2),int(hmax)),8,(255,0,255),-1)

	if show_images:
		cv2.imshow("Lines Found: RANSAC Before Modification", disp_lines)
		cv2.imshow("Lines Found: RANSAC After Modification", tmp)
		cv2.imshow("morphed",tmpFil)

	return tmp


def find_middle_line(line_left,line_right,limit=None,plot=False):
	rl,cl = line_left.shape
	rr,cr = line_right.shape
	midLine = []; midLine_coef = []

	try:
		nData = limit
		rng = range(nData)
	except:
		nData = min(cl,cr)
		rng = range(nData)

	# print("Data Limit: ", nData)
	# Could reduce this to just using the end points to save some time
	midx = [np.mean([line_left[0][i], line_right[0][i]]) for i in rng]
	midy = [np.mean([line_left[1][i], line_right[1][i]]) for i in rng]

	pts = np.array([ [midx[0], midy[0]],
					 [midx[-1], midy[-1]]
	])

	try:
		midLine_coef = np.polyfit(pts[:,1], pts[:,0], 1)
	except:
		print("ERROR: Could not find middle reference line coefficients")
		return False

	if plot:
		plot_ref = midLine_coef[0]*line_left[1][:] + midLine_coef[1]
		plt.plot(line_left[0,:], line_left[1,:],c='blue')
		plt.plot(line_right[0,:], line_right[1,:],c='red')
		plt.plot(midx, midy, '--', c='yellow')
		plt.plot(plot_ref, line_left[1][:], '+', c='green')
		plt.show()

	midLine = np.array([midx,midy])
	return midLine_coef, midLine, pts

def find_middle_reference_line(line_left,line_right, bottom_pixel):
	interPt = find_line_intersection(line_left, line_right)
	pts = np.array([ [interPt[0], interPt[1]],
					 [interPt[0],bottom_pixel]
	])

	try:
		line_fit = np.polyfit(pts[:,1], pts[:,0], 1)
		return line_fit,interPt, pts
	except:
		print("ERROR: Could not find middle reference line")
		return False

def find_line_angle(line_in_pts,line_ref_pts,verbose=True):
	angleIn = 0; angleRef = 0
	# print line_in_pts
	# print line_ref_pts

	x1r = int(line_ref_pts[0][0]); y1r = int(line_ref_pts[0][1])
	x2r = int(line_ref_pts[1][0]); y2r = int(line_ref_pts[1][1])

	x1i = int(line_in_pts[0][0]); y1i = int(line_in_pts[0][1])
	x2i = int(line_in_pts[1][0]); y2i = int(line_in_pts[1][1])

	inner_productr = x1r*x2r + y1r*y2r
	len1r = math.hypot(x1r, y1r)
	len2r = math.hypot(x2r, y2r)

	inner_producti = x1i*x2i + y1i*y2i
	len1i = math.hypot(x1i, y1i)
	len2i = math.hypot(x2i, y2i)

	if verbose == True:
		print "Reference Pts: ",x1r,y1r,x2r,y2r
		print "Input Pts: ", x1i,y1i,x2i,y2i

	x1 = (x2r - x1r); y1 = (y2r-y1r)
	x2 = (x2i - x1i); y2 = (y2i-y1i)
	inner_product = (x1*x2) + (y1*y2)
	len1 = math.hypot(x1, y1)
	len2 = math.hypot(x2, y2)

	try:
		angleTest = math.acos(inner_product/(len1*len2))
		angleTest_deg = (angleTest*180/math.pi)
	except:
		angleTest = 0
		print("ERROR: Could not find the angle of the middle reference line")

	if x2i < x1r:
		angle_out = -1*angleTest
	else:
		angle_out = angleTest
	# try:
	# 	angle = math.atan2((y1i-y2r),(x2i-x1r))
	# 	angle_deg = angle*180/math.pi
	# except:
	# 	angle = 0
	# 	print("ERROR")
	#
	# if angle < 0:
	# 	angle_out = (math.pi/2)+angle
	# else:
	# 	angle_out = (math.pi/2)-angle

	# print("Angle of the Middle Line (deg): ", angleTest_deg)
	return angle_out,x1r

def line(p1, p2):
	A = (p1[1] - p2[1])
	B = (p2[0] - p1[0])
	C = (p1[0]*p2[1] - p2[0]*p1[1])
	return A, B, -C

def intersection(L1, L2):
	D  = L1[0] * L2[1] - L1[1] * L2[0]
	Dx = L1[2] * L2[1] - L1[1] * L2[2]
	Dy = L1[0] * L2[2] - L1[2] * L2[0]
	if D != 0:
		x = Dx / D
		y = Dy / D
		return x,y
	else:
		return False

def find_line_intersection(line1,line2):
	x00 = line1[0][0]; y00 = line1[1][0]; x01 = line1[0][-1]; y01 = line1[1][-1]
	x10 = line2[0][0]; y10 = line2[1][0]; x11 = line2[0][-1]; y11 = line2[1][-1]

	L1 = line([x00,y00], [x01,y01])
	L2 = line([x10, y10], [x11,y11])

	R = intersection(L1, L2)
	if R:
		print "Intersection detected:", R
		return R
	else:
		print "No single intersection point detected"
		return False
