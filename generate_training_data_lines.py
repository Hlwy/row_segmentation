import os
import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt
from utils import utils as ut

# Mouse callback function for labeling
ix,iy = -1,-1
def draw_line_point(event,x,y,flags,param):
	global ix,iy

	if event == cv2.EVENT_LBUTTONDBLCLK:
		if flags == (cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_LBUTTON):
			cv2.circle(display,(x,y),5,(0,0,255),-1)
			ix,iy = x,y
			right_xs.append(x), right_ys.append(y)
			# print("Right Line: X\'s: " + str(right_xs) + " ----- Y\'s:" + str(right_ys))
		else:
			cv2.circle(display,(x,y),5,(255,0,0),-1)
			ix,iy = x,y
			left_xs.append(x), left_ys.append(y)
			# print("Left Line: X\'s: " + str(left_xs) + " ----- Y\'s:" + str(left_ys))


#Run Main
if __name__ == "__main__" :

	global left_xs,left_ys,right_xs,right_ys
	left_xs = []; left_ys = []
	right_xs = []; right_ys = []
	print('-------------------------------')
	# Setup commandline argument(s) structures
	ap = argparse.ArgumentParser(description='Image Segmentation')
	ap.add_argument("--images_path", "-i", type=str, metavar='FILE', default='test', help="Name of video file to parse")
	ap.add_argument("--output_file", "-n", type=str, metavar='NAME', default='training_log', help="Name of output file containing information about parsed video")
	ap.add_argument("--output_path", "-p", type=str, metavar='FILE', default='exported', help="Name of video file to parse")
	ap.add_argument("--input_log", "-l", type=str, metavar='FILE', help="Name of video file to parse")
	# Store parsed arguments into array of variables
	args = vars(ap.parse_args())

	# Extract stored arguments array into individual variables for later usage in script
	imgDir = args["images_path"]
	outName = args["output_file"]
	outDir = args["output_path"]
	inputLog = args["input_log"]

	outName = str(imgDir) + "_" + str(outName)
	# outDir = os.getcwd() + "/" + str(outDir)
	print("	Output Directory:			" + os.getcwd() + "/" + str(outDir))

	# Store initial image variables
	_imgs, _paths = ut.get_images_by_dir(imgDir)
	img = _imgs[0]
	new_img_path = _paths[0]
	cur_img = cv2.resize(img, (640,480))
	clone = cv2.resize(img, (640,480))
	display = cv2.resize(img, (640,480))

	# Graphical Entities
	cv2.namedWindow('image')
	cv2.setMouseCallback('image',draw_line_point)
	plt.ion()

	# Initialize miscellaneous parameters
	i = 0; n = len(_imgs); count_record = 0
	# csvHeaders = ["Local Image Path", "Slope - Left", "Intercept - Left", "Slope - Right", "Intercept - Right"]
	csvHeaders = ["image", "mLeft", "bLeft", "mRight", "bRight"]
	csvList = []

	# Random variables used for quick debugging
	h,w,c = cur_img.shape
	ploty = np.linspace(0, w-1, w)

	while True:
		cv2.imshow('image', display)
		key = cv2.waitKey(5) & 0xFF
		flag_recorded = False

		# Parse the user-input
		if key == ord('r'):
			left_fit = np.polyfit(left_ys, left_xs, 1)
			right_fit = np.polyfit(right_ys, right_xs, 1)
			plot_leftx = left_fit[0]*ploty + left_fit[1]
			plot_rightx = right_fit[0]*ploty + right_fit[1]
			cv2.line(clone,(int(plot_leftx[0]),int(ploty[0])),(int(plot_leftx[-1]),int(ploty[-1])),(0,0,255))
			cv2.line(clone,(int(plot_rightx[0]),int(ploty[0])),(int(plot_rightx[-1]),int(ploty[-1])),(255,0,0))

			plt.imshow(clone)
			plt.title(str(new_path))

			tmpData = [new_path, left_fit[0], left_fit[1], right_fit[0], right_fit[1]]
			csvList.append(tmpData)
			flag_recorded = True
			count_record += 1
			# print("----- Storing Currently Chosen Parameters....")
			print("Data Recording ----- " + str(count_record) + " entries recorded")
		if key == ord('c'):
			left_xs = []
			left_ys = []
			right_xs = []
			right_ys = []
			display = np.copy(cur_img)
			print("Clearing current line parameters:")#  X\'s: " + str(xs) + " .... Y\'s:" + str(ys))
		if key == ord('s'):
			print("----- Exporting saved training data to csv file....")
			ut.export_list2csv(outDir, outName, csvHeaders, csvList)
		if key == ord('q'):
			break

		new_img, new_path, i, flag_new_img = ut.cycle_through_images(key, _imgs, _paths, i, [flag_recorded])

		if flag_new_img == True:
			left_xs = []
			left_ys = []
			right_xs = []
			right_ys = []
			cur_img = cv2.resize(new_img, (640,480))
			clone = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
			display = cv2.resize(new_img, (640,480))

			cv2.imshow('image', display)

		plt.show()
		plt.pause(0.01)

	cv2.destroyAllWindows()
