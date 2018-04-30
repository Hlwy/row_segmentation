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


def find_line(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
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

	tmp, M, Minv = zhong_warp(img, _alpha,_beta,_gamma, _focal,_dist)

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

def zhong_warp(_img, _alpha=45.0, _beta=0.0,_gamma=0.0, _focal=500.0, _dist=500.0):
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

def warp_perspective(_img):

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
