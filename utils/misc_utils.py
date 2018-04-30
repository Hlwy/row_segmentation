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


def strip_process(image_edit, n_strips=10):

	height = len(image_edit)
	width = len(image_edit[0])

	strip_height = height / n_strips
	crop_points = np.zeros((height, width), dtype=np.uint8)

	strips = [image_edit[(strip_number*strip_height):((strip_number+1)*strip_height-1), :] for strip_number in range(n_strips)]
	for strip_number in range(n_strips):
		# image_strip = image_edit[(strip_number*strip_height):((strip_number+1)*strip_height-1), :]
		cv2.imshow(str(strip_number), strips[strip_number])

	return strips

def sliding_window():
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
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2)
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2)
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

def draw_area(undist,binary_warped,Minv,left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def calculate_curv_and_pos(binary_warped,left_fit, right_fit):
    # Define y-value where we want radius of curvature
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
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
    return curvature,distance_from_center



def draw_values(img,curvature,distance_from_center):
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius_text = "Radius of Curvature: %sm"%(round(curvature))

    if distance_from_center>0:
        pos_flag = 'right'
    else:
        pos_flag= 'left'

    cv2.putText(img,radius_text,(100,100), font, 1,(255,255,255),2)
    center_text = "Vehicle is %.3fm %s of center"%(abs(distance_from_center),pos_flag)
    cv2.putText(img,center_text,(100,150), font, 1,(255,255,255),2)
    return img


def processing(img,object_points,img_points,M,Minv,left_line,right_line):
    undist = cal_undistort(img,object_points,img_points)
    thresholded = thresholding(undist)
    thresholded_wraped = cv2.warpPerspective(thresholded, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    if left_line.detected and right_line.detected:
        left_fit, right_fit, left_lane_inds, right_lane_inds = find_line_by_previous(thresholded_wraped,left_line.current_fit,right_line.current_fit)
    else:
        left_fit, right_fit, left_lane_inds, right_lane_inds = find_line(thresholded_wraped)
    left_line.update(left_fit)
    right_line.update(right_fit)

    area_img = draw_area(undist,thresholded_wraped,Minv,left_fit, right_fit)

    curvature,pos_from_center = calculate_curv_and_pos(thresholded_wraped,left_fit, right_fit)
    result = draw_values(area_img,curvature,pos_from_center)
    return result
