#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

#include "filters/color_filter.h"

using namespace std;

int h = 480; int w = 640;

int hsv_ll[] = {32, 52, 0};		// Alright Case
int hsv_ul[] = {107, 255, 255};	// Alright Case
// int hsv_ll[] = {18, 31, 25};	// Test Case
// int hsv_ul[] = {111, 255, 130}; // Test Case

int yuv_ll[] = {0, 0, 0};		// Original Case
int yuv_ul[] = {164, 126, 126};	// Original Case
// int yuv_ll[] = {72, 107, 107};	// Test Case
// int yuv_ul[] = {148, 129, 135};	// Test Case


/**  @function main */
int main(int argc, char** argv){
	// OS Variables
	char* default_img_path = "/home/hunter/data/segmentation/farm/testing/11.jpg";
	char* img_path;
	// Labels
	string source_window = "Source Image";

	// Image Container Initializations
	cv::Mat src;		// Raw, Unknown Sized, Source Image (BGR Color Space)
	cv::Mat ssrc; 		// Raw Source Image resized to defined size standard
	cv::Mat cfilled; 	// Image resulting from combined color filter pre-processing

	// Object Declarations
	ColorFilter hsvFilter(HSV);
	hsvFilter.set_upper_limits(hsv_ul);
	hsvFilter.set_lower_limits(hsv_ll);
	ColorFilter yuvFilter(HSV);
	yuvFilter.set_upper_limits(yuv_ul);
	yuvFilter.set_lower_limits(yuv_ll);

	// Command-line Parsing
	if(argc < 2){
		cout << "Usage: " << argv[0] << " <path_to_image>\r\n"
			<< "	Using default image path " << default_img_path << endl;
		img_path = default_img_path;
	}else img_path = argv[1];

	// Load an image
	src = cv::imread(img_path,1);

	// Double check if source image stored properly and if so resize to standard
	if(src.empty()){
		cerr << "Loading Source image failed. (No loaded data)" << endl;
		return -1;
	}else{
		cv::resize(src,ssrc,cv::Size(w,h));
	}

	/** ---------------------------------------------------------
	*		Filter out specific colors from source image
	* --------------------------------------------------------- */
	hsvFilter.print_internals();
	cfilled = hsvFilter.filter_color(ssrc);
	cv::imshow("HSV Filtered Image",cfilled);

	yuvFilter.print_internals();
	cfilled = yuvFilter.filter_color(ssrc);
	cv::imshow("YUV Filtered Image",cfilled);

	// 		_, mask_hsv = cv2.threshold(mask_hsv, 10, 255, cv2.THRESH_BINARY)
	// 	res_hsv = cv2.bitwise_and(tmp, tmp, mask = mask_hsv)
	//
	// 	comp_mask = cv2.bitwise_and(mask_yuv,mask_hsv)
	// 	_, comp_mask = cv2.threshold(comp_mask, 10, 255, cv2.THRESH_BINARY)
	// 	res = cv2.bitwise_and(tmp, tmp, mask = comp_mask)
	//
	// 	return res, comp_mask

	/** -------------------------------------------------------------------
	*		Apply Morphological Transforms on color filtered image
	* --------------------------------------------------------------------- */
	// def apply_morph(_img, ks=[5,5], shape=0, flag_open=False, flag_show=True):
	// 	if shape == 0:
	// 		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(int(ks[0]),int(ks[1])))
	// 	elif shape == 1:
	// 		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(ks[0]),int(ks[1])))
	// 	else:
	// 		print("alternative structures here...")
	//
	// 	blurred = cv2.medianBlur(_img, 7)
	// 	opening = cv2.morphologyEx(blurred,cv2.MORPH_OPEN,kernel)
	// 	closing = cv2.morphologyEx(blurred,cv2.MORPH_CLOSE,kernel)
	// 	if flag_show == True:
	// 		cv2.imshow('Before Morphing',_img)
	// 		# cv2.imshow('Blurred',blurred)
	// 		cv2.imshow('opened',opening)
	// 		cv2.imshow('closed',closing)
	//
	// 	if flag_open == True:
	// 		out = opening
	// 	else:
	// 		out = closing
	// 	return out


	// Show Image
	cv::namedWindow(source_window, CV_WINDOW_NORMAL);
	cv::imshow(source_window, src);
	// Wait until user exits the program
	cv::waitKey(0);

	return 0;
}
