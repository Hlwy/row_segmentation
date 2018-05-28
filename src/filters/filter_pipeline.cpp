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

cv::Mat FilterPipeline::blur_filtered(const cv::Mat& src, int aperture,bool show){
	cv::Mat blurred = src.clone();
	// Ensure inputs are good
	if(aperture % 2 == 0) aperture = 3;
	// Apply Blurring Effects
	cv::medianBlur(src, blurred,aperture);
	if(show){
		string lblSrc = "FilterPipeline: Blurring Input Image";
		string lblBlur = "FilterPipeline: Blurred Image";
		cv::namedWindow(lblSrc, CV_WINDOW_NORMAL);
		cv::namedWindow(lblBlur, CV_WINDOW_NORMAL);
		cv::imshow(lblSrc,src);
		cv::imshow(lblBlur,blurred);
	}
	return blurred;
}

cv::Mat FilterPipeline::morph_filtered(const cv::Mat& src, uint8_t kernel_size[2], MorphElement shape, MorphType morphing, bool show){
	cv::Mat element, morphed;
	cv::Mat clone = src.clone();
	// Variables to check for input validity
	MorphElement elems = num_elements;
	MorphType types = num_types;

	// If invalid morphing type default to Opening
	if(morphing > num_types) morphing = OPEN;
	// Ensure a valid structuring element is used, if not use a default
	if(shape > elems){
		shape = RECTANGLE;
		element = cv::getStructuringElement(shape, cv::Size(kernel_size[0], kernel_size[1]));
	}else element = cv::getStructuringElement(shape, cv::Size(kernel_size[0], kernel_size[1]));

	// Perform Morphological Transformation
	if(morphing == OPEN) cv::morphologyEx(clone, morphed, cv::MORPH_OPEN, element);
	else if(morphing == CLOSE) cv::morphologyEx(clone, morphed, cv::MORPH_CLOSE, element);
	else if(morphing == ERODE) cv::erode(clone, morphed, element);
	else if(morphing == DILATE) cv::dilate(clone, morphed, element);
	else morphed = clone;

	if(show){
		string lblSrc = "FilterPipeline: Morphing Input Image";
		string lblMorph = "FilterPipeline: Morphed Image";
		cv::namedWindow(lblSrc, CV_WINDOW_NORMAL);
		cv::namedWindow(lblMorph, CV_WINDOW_NORMAL);
		cv::imshow(lblSrc,clone);
		cv::imshow(lblMorph,morphed);
	}
	return morphed;
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
	cv::Mat tmp, mask, filtered, result, combined_mask;
	vector<cv::Mat> tmps;
	vector<cv::Mat> masks;
	uint8_t ksize[] = {5,5};

	// TODO: Dynamically handle stored ColorFilter's
	ColorFilter* yuv = this->filters.at(0);
	ColorFilter* hsv = this->filters.at(1);

	// Filter: YUV Color space
	yuv->filter_color(src);
	tmps.push_back(yuv->filtered);
	masks.push_back(yuv->mask);

	// Filter: HSV Color space
	hsv->filter_color(src);
	tmps.push_back(hsv->filtered);
	masks.push_back(hsv->mask);

	/** NOTE:
	*		Original python implementation of row_segmentation used
	*	thresholding of the masks before combining them, like so:
	*
	*	_, mask_hsv = cv2.threshold(mask_hsv, 10, 255, cv2.THRESH_BINARY)
	*	res_hsv = cv2.bitwise_and(tmp, tmp, mask = mask_hsv)
	*
	*	Not sure if this has any significant effect. POSSIBLY LOOK INTO LATER.
	*/
	int nCspaces = 2;

	// Combined the two filtered masks
	for(int i = 0; i<nCspaces-1;i++){
		cout << "Combined Filter Masks: " << i << endl;
		// If just starting create the first instance of the combined filter mask
		if(i == 0){
			cv::bitwise_and(masks.at(i), masks.at(i+1),combined_mask);
			i++;
		} else cv::bitwise_and(combined_mask, masks.at(i),combined_mask);
	}

	// Apply combined mask to source image
 	cv::bitwise_and(src, src, filtered, combined_mask);

	// Blur the Resultant image
	cv::Mat blurred = this->blur_filtered(filtered);

	// Apply morphological transformations. TODO: Look into various combinations?
	cv::Mat morphed = this->morph_filtered(blurred,ksize,ELLIPSE,DILATE);

	if(show){
		string lblSrc = "FilterPipeline: Source Image";
		string lblMask = "FilterPipeline: Combined Mask";
		string lblRes = "FilterPipeline: Color Filtered";
		string lblBlur = "FilterPipeline: Blurred";
		string lblMorph = "FilterPipeline: Morphed";
		cv::namedWindow(lblSrc, CV_WINDOW_NORMAL);
		cv::namedWindow(lblMask, CV_WINDOW_NORMAL);
		cv::namedWindow(lblRes, CV_WINDOW_NORMAL);
		cv::namedWindow(lblBlur, CV_WINDOW_NORMAL);
		cv::namedWindow(lblMorph, CV_WINDOW_NORMAL);
		cv::imshow(lblSrc,src);
		cv::imshow(lblMask,combined_mask);
		cv::imshow(lblRes,filtered);
		cv::imshow(lblBlur,blurred);
		cv::imshow(lblMorph,morphed);
	}

	return morphed;
}


// Get Functions
ucube FilterPipeline::get_colorspace_limits(){return this->_limits;}
int FilterPipeline::get_colorspace_count(){return this->_ccount;}

// Debugging Functions
void FilterPipeline::print_internals(){
	cout << on_magenta << "FilterPipeline:" << reset << endl
		<< "	Color Space Used: " << this->_id << endl;
}
