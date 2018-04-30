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
import os
import csv
import argparse
from matplotlib import pyplot as plt
import numpy as np
import utils


def onChange(pos):
    global img
    global _imgs
    global tmp
    global i

    img = _imgs[i]
    # tmp = np.copy(img)
    tmp = cv2.resize(img, (640,480))

    # get current positions of four trackbars
    hmin = cv2.getTrackbarPos('Hmin','image')
    smin = cv2.getTrackbarPos('Smin','image')
    vmin = cv2.getTrackbarPos('Vmin','image')
    hmax = cv2.getTrackbarPos('Hmax','image')
    smax = cv2.getTrackbarPos('Smax','image')
    vmax = cv2.getTrackbarPos('Vmax','image')
    ymin = cv2.getTrackbarPos('Ymin','image')
    umin = cv2.getTrackbarPos('Umin','image')
    vvmin = cv2.getTrackbarPos('VVmin','image')
    ymax = cv2.getTrackbarPos('Ymax','image')
    umax = cv2.getTrackbarPos('Umax','image')
    vvmax = cv2.getTrackbarPos('VVmax','image')
    ks = cv2.getTrackbarPos('kernelSize','image')
    ws = cv2.getTrackbarPos('colorSize','image')
    thresh_flag = cv2.getTrackbarPos('threshInv','image')

    gray = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(tmp,cv2.COLOR_BGR2HSV)
    yuv = cv2.cvtColor(tmp,cv2.COLOR_BGR2YUV)

    lower_yuv = np.array([ymin, umin, vvmin])
    upper_yuv = np.array([ymax, umax, vvmax])

    # lower_yuv = np.array([107, 0, 0])
    # upper_yuv = np.array([255, 116, 255])

    mask1 = cv2.inRange(yuv, lower_yuv, upper_yuv)
    res1 = cv2.bitwise_and(tmp, tmp, mask = mask1)
    # plt.imshow(res, cmap='gray')
    cv2.imshow('white filtered',res1)


    # lower_red2 = np.array([hmax, smax, vmax])
    # upper_red2 = np.array([255, 255, 255])

    lower_yuv2 = np.array([hmin, smin, vmin])
    upper_yuv2 = np.array([hmax, smax, vmax])

    mask2 = cv2.inRange(hsv, lower_yuv2, upper_yuv2)
    res2 = cv2.bitwise_and(tmp, tmp, mask = mask2)
    cv2.imshow('color filtered',res2)

    if thresh_flag == 0:
        comp_mask = mask1 | mask2
    if thresh_flag == 1:
        comp_mask = mask1 & mask2

    res = cv2.bitwise_and(tmp, tmp, mask = comp_mask)

    cv2.imshow('Composite mask',comp_mask)
    cv2.imshow('Composite filtered',res)

    gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

    if thresh_flag == 0:
        ret, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    if thresh_flag == 1:
        ret, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    res3 = cv2.bitwise_and(tmp, tmp, mask = thresh)

    # test = cv2.pyrMeanShiftFiltering(res,ks, ws, 3)
    # cv2.imshow('Composite filtered Blurred',test)
	
    # kernel = np.ones((ks,ks),np.uint8)
    # opening = cv2.morphologyEx(res3,cv2.MORPH_OPEN,kernel, iterations = 2)
    # cv2.imshow('opened',opening)
    # closing = cv2.morphologyEx(res3,cv2.MORPH_CLOSE,kernel, iterations = 2)
    # cv2.imshow('closed',closing)
    #
    # if thresh_flag == 0:
    #     gray2 = cv2.cvtColor(opening,cv2.COLOR_BGR2GRAY)
    # if thresh_flag == 1:
    #     gray2 = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)
    #
    # ret, helper = cv2.threshold(gray2,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # hlines = helper
    # lines = cv2.HoughLinesP(hlines,1,np.pi/180,desiredThresh,minLineLength,maxLineGap)
    # for x in range(0, len(lines)):
    #     for x1,y1,x2,y2 in lines[x]:
    #         cv2.line(tmp,(x1,y1),(x2,y2),(0,255,0),2)


#Run Main
if __name__ == "__main__" :

    # Setup commandline argument(s) structures
    ap = argparse.ArgumentParser(description='Image Segmentation')
    ap.add_argument("--pic", "-p", type=str, default='test_imgs', metavar='FILE', help="Name of video file to parse")
    # Store parsed arguments into array of variables
    args = vars(ap.parse_args())

    # Extract stored arguments array into individual variables for later usage in script
    _img = args["pic"]
    _imgs = utils.get_images_by_dir(_img)

    # create trackbars for color change
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('Hmin','image',0,255,onChange)
    cv2.createTrackbar('Smin','image',0,255,onChange)
    cv2.createTrackbar('Vmin','image',0,255,onChange)
    cv2.createTrackbar('Hmax','image',255,255,onChange)
    cv2.createTrackbar('Smax','image',24,255,onChange)
    cv2.createTrackbar('Vmax','image',232,255,onChange)
    cv2.createTrackbar('Ymin','image',39,255,onChange)
    cv2.createTrackbar('Umin','image',124,255,onChange)
    cv2.createTrackbar('VVmin','image',106,255,onChange)
    cv2.createTrackbar('Ymax','image',202,255,onChange)
    cv2.createTrackbar('Umax','image',255,255,onChange)
    cv2.createTrackbar('VVmax','image',129,255,onChange)
    cv2.createTrackbar('kernelSize','image',0,255,onChange)
    cv2.createTrackbar('colorSize','image',0,255,onChange)
    cv2.createTrackbar('threshInv','image',0,1,onChange)

    img = _imgs[0]	# Input video file as OpenCV VideoCapture device
    tmp = cv2.resize(img, (640,480))
    i = 0

    n = len(_imgs)
    # print n

    while True:
        # print i
        tmp = cv2.resize(_imgs[i], (640,480))
        cv2.imshow("image", tmp)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            i = i + 1
            if i >= n:
                i = 0
            print 'Next Image...'

    cv2.destroyAllWindows()
