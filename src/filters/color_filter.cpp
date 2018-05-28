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

void ColorFilter::filter_color(const cv::Mat& src, bool show){
	cv::Mat cmap_image, mask, cmap_filtered, result;

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
		string lblSrc = "ColorFilter: [" + string(this->_lbl) + "] Color Spaced Image";
		string lblMask = "ColorFilter: [" + string(this->_lbl) + "] Mask";
		string lblRes = "ColorFilter: [" + string(this->_lbl) + "] Result";
		cv::namedWindow(lblSrc, CV_WINDOW_NORMAL);
		cv::imshow(lblSrc,cmap_image);
		cv::namedWindow(lblMask, CV_WINDOW_NORMAL);
		cv::imshow(lblMask,mask);
		cv::namedWindow(lblRes, CV_WINDOW_NORMAL);
		cv::imshow(lblRes,result);
	}

	// Store/return important variables
	this->mask = mask.clone();
	this->filtered = result.clone();
}

// Set Functions
void ColorFilter::set_colorspace(ColorSpace cmap){
	ColorSpace maps = num_spaces;
	if(cmap == HSV){
		this->_cmap = cmap;
		this->_lbl = "HSV";
	}else if(cmap == YUV){
		this->_cmap = cmap;
		this->_lbl = "YUV";
	}else if(cmap == RGB){
		this->_cmap = cmap;
		this->_lbl = "RGB";
	}else{
		this->_cmap = BGR;
		this->_lbl = "BGR";
	}
}

void ColorFilter::set_upper_limits(vector<uint8_t> limits,bool verbose){
	// Check for errorneous limits
	if(limits[0] > 255) this->_upper_limits[0] = 255;
	else this->_upper_limits[0] = limits[0];

	if(limits[1] > 255) this->_upper_limits[1] = 255;
	else this->_upper_limits[1] = limits[1];

	if(limits[2] > 255) this->_upper_limits[2] = 255;
	else this->_upper_limits[2] = limits[2];

	// Debugging
	if(verbose == true){
		cout << yellow << bold << "ColorFilter [set_upper_limits]:" << reset << endl
		<< yellow << "	Input Limits: " << reset << to_string(limits[0]) << ", " << to_string(limits[1]) << ", " << to_string(limits[2]) << endl
		<< yellow << "	Set Lower Limits: " << reset << to_string(this->_upper_limits[0]) << ", " << to_string(this->_upper_limits[1]) << ", " << to_string(this->_upper_limits[2]) << endl;
	}
}

void ColorFilter::set_lower_limits(vector<uint8_t> limits,bool verbose){
	// Check for errorneous limits
	if(limits[0] < 0) this->_lower_limits[0] = 0;
	else this->_lower_limits[0] = limits[0];

	if(limits[1] < 0) this->_lower_limits[1] = 0;
	else this->_lower_limits[1] = limits[1];

	if(limits[2] < 0) this->_lower_limits[2] = 0;
	else this->_lower_limits[2] = limits[2];

	// Debugging
	if(verbose == true){
		cout << yellow << bold << "ColorFilter [set_lower_limits]:" << reset << endl
			<< yellow << "	Input Limits: " << reset << to_string(limits[0]) << ", " << to_string(limits[1]) << ", " << to_string(limits[2]) << endl
			<< yellow << "	Set Lower Limits: " << reset << to_string(this->_lower_limits[0]) << ", " << to_string(this->_lower_limits[1]) << ", " << to_string(this->_lower_limits[2]) << endl;
	}
}

// Get Functions
vector<uint8_t> ColorFilter::get_upper_limits(){return this->_upper_limits;}
vector<uint8_t> ColorFilter::get_lower_limits(){return this->_lower_limits;}

// Debugging Functions
void ColorFilter::print_internals(){
	// Name contractions for easier printing
	vector<uint8_t> ll = this->_lower_limits;
	vector<uint8_t> ul = this->_upper_limits;

	cout << on_magenta << "ColorFilter:" << reset << endl
		<< "	Color Space Used: " << this->_lbl << endl
		<< "	Lower Limits: " << to_string(ll[0]) << ", " << to_string(ll[1]) << ", " << to_string(ll[2]) << endl
		<< "	Upper Limits: " << to_string(ul[0]) << ", " << to_string(ul[1]) << ", " << to_string(ul[2])  << endl;

}

// Debugging Functions
void ColorFilter::print_internals(string label){
	// Name contractions for easier printing
	vector<uint8_t> ll = this->_lower_limits;
	vector<uint8_t> ul = this->_upper_limits;

	cout << on_magenta << label << reset << endl
		<< "	Color Space Used: " << this->_lbl << endl
		<< "	Lower Limits: " << to_string(ll[0]) << ", " << to_string(ll[1]) << ", " << to_string(ll[2]) << endl
		<< "	Upper Limits: " << to_string(ul[0]) << ", " << to_string(ul[1]) << ", " << to_string(ul[2])  << endl;

}
