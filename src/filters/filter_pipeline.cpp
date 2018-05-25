#include <iostream>
#include <stdio.h>
#include <termcolor/termcolor.hpp>

#include "filters/filter_pipeline.h"

using namespace std;
using namespace termcolor;
using namespace arma;

FilterPipeline::FilterPipeline(){cout << "Filter Pipeline: Initialized\r\n";}

FilterPipeline::FilterPipeline(ColorSpace cmaps[], umat limits, bool verbose){
	int cmap_sz = sizeof(cmaps)/sizeof(ColorSpace);
	// Load all filters
	for(int i = 0;i<cmap_sz;i++){	this->add_color_filter(cmaps[i],limits.row(i), verbose);}
}

FilterPipeline::~FilterPipeline(){
	for(int i = 0; i==this->_ccount;i++){ delete this->filters.at(i); }
}

// Set Functions
void FilterPipeline::add_color_filter(ColorSpace cmap, umat limits, bool verbose){
	// Initialize an empty ColorFilter Object to be loaded with parameters
	ColorSpace maps = num_spaces;
	int index = this->_ccount;

	// Store string identifiers of specific Color filters for external debugging
	if(cmap == HSV)this->_lbls.push_back("HSV");
	else if(cmap == YUV) this->_lbls.push_back("YUV");
	else if(cmap == RGB) this->_lbls.push_back("RGB");
	else if(cmap == BGR) this->_lbls.push_back("BGR");
	else cmap = BGR;

	ColorFilter* tmpFilter = new ColorFilter(cmap);
	this->filters.push_back(tmpFilter);

	// Add empty row for setting color space limits
	umat filler(2,3, fill::zeros);
	// Store Filter-specific limits to a global limit container for easier debugging
	this->_limits = join_slices(this->_limits,filler);
	this->set_color_filter_limits(index,limits,verbose);

	string desc = "ColorFilter [" + to_string(index) + "]: ";
	this->filters.at(index)->print_internals(desc);

	this->_ccount++;
}

void FilterPipeline::set_color_filter_limits(int index, umat limits, bool verbose){
	umat satLimits(2,3, fill::zeros);
	// Check for errorneous lower limits
	if(as_scalar(limits(0,0)) < 0) satLimits(0,0) = 0;
	else satLimits(0,0) = as_scalar(limits(0,0));

	if(as_scalar(limits(0,1)) < 0) satLimits(0,1) = 0;
	else satLimits(0,1) = as_scalar(limits(0,1));

	if(as_scalar(limits(0,2)) < 0) satLimits(0,2) = 0;
	else satLimits(0,2) = as_scalar(limits(0,2));
	// Check for errorneous upper limits
	if(as_scalar(limits(0,3)) > 255) satLimits(1,0) = 255;
	else satLimits(1,0) = as_scalar(limits(0,3));

	if(as_scalar(limits(0,4)) > 255) satLimits(1,1) = 255;
	else satLimits(1,1) = as_scalar(limits(0,4));

	if(as_scalar(limits(0,5)) > 255) satLimits(1,2) = 255;
	else satLimits(1,2) = as_scalar(limits(0,5));

	satLimits.row(0) = limits.cols(0,2);
	satLimits.row(1) = limits.cols(3,5);
	this->_limits.slice(index) = satLimits;
	this->filters.at(index)->set_lower_limits(conv_to< vector<uint8_t> >::from(satLimits.row(0)), verbose);
	this->filters.at(index)->set_upper_limits(conv_to< vector<uint8_t> >::from(satLimits.row(1)), verbose);
}

/** ---------------------------------------------------------
*		Filter out specific colors from source image
* --------------------------------------------------------- */
cv::Mat FilterPipeline::filter_image(const cv::Mat& src, bool show){
	cv::Mat tmp, mask, result, combined_mask;
	vector<cv::Mat> tmps;
	vector<cv::Mat> masks;

	// TODO: Dynamically handle stored ColorFilter's
	ColorFilter* yuv = this->filters.at(0);
	ColorFilter* hsv = this->filters.at(1);

	// Filter: YUV Color space
	yuv->filter_color(src,show);
	tmps.push_back(yuv->filtered);
	masks.push_back(yuv->mask);

	// Filter: HSV Color space
	hsv->filter_color(src,show);
	tmps.push_back(hsv->filtered);
	masks.push_back(hsv->mask);

	/** NOTE:
	*		Original python implementation of this used thresholding of the
	*	masks before combining them, like so:
	*
	*	_, mask_hsv = cv2.threshold(mask_hsv, 10, 255, cv2.THRESH_BINARY)
	*	res_hsv = cv2.bitwise_and(tmp, tmp, mask = mask_hsv)
	*
	*	Not sureif this has any significant effect. POSSIBLY LOOK INTO LATER.
	*/
	int nImgs = 2;

	// Combined the two filtered masks
	for(int i = 0; i<nImgs-1;i++){
		cout << "Combined Filter Masks: " << i << endl;
		// If just starting create the first instance of the combined filter mask
		if(i == 0){
			cv::bitwise_and(masks.at(i), masks.at(i+1),combined_mask);
			i++;
		} else cv::bitwise_and(combined_mask, masks.at(i),combined_mask);
	}

	// Apply combined mask to source image
 	cv::bitwise_and(src, src, result, combined_mask);

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

	if(show){
		string lblSrc = "FilterPipeline: Source Image";
		string lblMask = "FilterPipeline: Combined Mask";
		string lblRes = "FilterPipeline: Result";
		cv::namedWindow(lblSrc, CV_WINDOW_NORMAL);
		cv::imshow(lblSrc,src);
		cv::namedWindow(lblMask, CV_WINDOW_NORMAL);
		cv::imshow(lblMask,combined_mask);
		cv::namedWindow(lblRes, CV_WINDOW_NORMAL);
		cv::imshow(lblRes,result);
	}

	return result;
}


// Get Functions
ucube FilterPipeline::get_colorspace_limits(){return this->_limits;}
int FilterPipeline::get_colorspace_count(){return this->_ccount;}

// Debugging Functions
void FilterPipeline::print_internals(){
	cout << on_magenta << "FilterPipeline:" << reset << endl
		<< "	Color Space Used: " << this->_id << endl;
}
