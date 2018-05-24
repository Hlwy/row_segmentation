#include <iostream>
#include <stdio.h>
#include <termcolor/termcolor.hpp>

#include "filters/color_filter.h"

using namespace std;
using namespace termcolor;

ColorFilter::ColorFilter(){}
ColorFilter::~ColorFilter(){}

ColorFilter::ColorFilter(ColorSpace cmap){
	this->set_colorspace(cmap);
}

cv::Mat ColorFilter::filter_image(const cv::Mat& src, bool show){
	cv::Mat cmap_image, mask, cmap_filtered, result;
	string lblSrc = "ColorFilter: [" + string(this->_space_desc) + "] Color Spaced Image";
	string lblMask = "ColorFilter: [" + string(this->_space_desc) + "] Mask";
	string lblRes = "ColorFilter: [" + string(this->_space_desc) + "] Result";

	// Convert Raw Source Image (BGR) into Target Color Space
	if(this->_cmap == HSV) cv::cvtColor(src, cmap_image, CV_BGR2HSV);
	else if(this->_cmap == YUV) cv::cvtColor(src, cmap_image, CV_BGR2YUV);
	else if(this->_cmap == RGB) cv::cvtColor(src, cmap_image, CV_BGR2RGB);
	else if(this->_cmap == BGR) cmap_image = src.clone();

	// Filter color space image  using internal limits
	cv::inRange(cmap_image, this->_lower_limits, this->_upper_limits, mask);
	cv::bitwise_and(cmap_image,cmap_image,cmap_filtered,mask);

	// Convert Color Space Filtered Image (Target) back into Original Color Space (BGR)
	if(this->_cmap == HSV) cv::cvtColor(cmap_filtered, result, CV_HSV2BGR);
	else if(this->_cmap == YUV) cv::cvtColor(cmap_filtered, result, CV_YUV2BGR);
	else if(this->_cmap == RGB) cv::cvtColor(cmap_filtered, result, CV_RGB2BGR);
	else if(this->_cmap == BGR) result = src.clone();

	if(show == true){
		cv::namedWindow(lblSrc, CV_WINDOW_NORMAL);
		cv::imshow(lblSrc,cmap_image);
		cv::namedWindow(lblMask, CV_WINDOW_NORMAL);
		cv::imshow(lblMask,mask);
		cv::namedWindow(lblRes, CV_WINDOW_NORMAL);
		cv::imshow(lblRes,result);
	}

	// Store/return important variables
	this->_mask = mask;
	return result;
}

// Set Functions
void ColorFilter::set_colorspace(ColorSpace cmap){
	ColorSpace maps = num_spaces;
	if(cmap == HSV){
		this->_cmap = cmap;
		this->_space_desc = "HSV";
	}else if(cmap == YUV){
		this->_cmap = cmap;
		this->_space_desc = "YUV";
	}else if(cmap == RGB){
		this->_cmap = cmap;
		this->_space_desc = "RGB";
	}else{
		this->_cmap = BGR;
		this->_space_desc = "BGR";
	}
}

void ColorFilter::set_upper_limits(int32_t limits[3]){
	// Check for errorneous limits
	if(limits[0] > 255) this->_upper_limits[0] = 255;
	else this->_upper_limits[0] = limits[0];

	if(limits[1] > 255) this->_upper_limits[1] = 255;
	else this->_upper_limits[1] = limits[1];

	if(limits[2] > 255) this->_upper_limits[2] = 255;
	else this->_upper_limits[2] = limits[2];
}

void ColorFilter::set_lower_limits(int32_t limits[3]){
	// Check for errorneous limits
	if(limits[0] < 0) this->_lower_limits[0] = 0;
	else this->_lower_limits[0] = limits[0];

	if(limits[1] < 0) this->_lower_limits[1] = 0;
	else this->_lower_limits[1] = limits[1];

	if(limits[2] < 0) this->_lower_limits[2] = 0;
	else this->_lower_limits[2] = limits[2];
}

// Get Functions
vector<int32_t> ColorFilter::get_upper_limits(){return this->_upper_limits;}
vector<int32_t> ColorFilter::get_lower_limits(){return this->_lower_limits;}

// Debugging Functions
void ColorFilter::print_internals(){
	// Name contractions for easier printing
	vector<int32_t> ll = this->_lower_limits;
	vector<int32_t> ul = this->_upper_limits;

	cout << on_magenta << "ColorFilter:" << reset << endl
		<< "	Color Space Used: " << this->_space_desc << endl
		<< "	Lower Limits: " << ll[0] << ", " << ll[1] << ", " << ll[2] << endl
		<< "	Upper Limits: " << ul[0] << ", " << ul[1] << ", " << ul[2]  << endl;

}
