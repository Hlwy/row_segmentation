#ifndef FILTER_PIPELINE_H_
#define FILTER_PIPELINE_H_

#include "color_filter.h"

using namespace std;

class FilterPipeline{
private:
	vector<ColorSpace> _cmaps;	// Color space that color filter is operating in
	vector<string> _lbls;		// Color Space identifiers for debugging
	const cv::Mat _filtered;		// Color Filtered Image of source image
	const cv::Mat _mask;		// Mask resultanting from color filtering

	// Upper and Lower Thresholding Limits per channel
	cv::Mat _upper_limits;
	cv::Mat _lower_limits;
public:
	/**
		Class Constructors/Deconstructors and Overloads
	*/
	FilterPipeline();
	FilterPipeline(ColorSpace cmaps[]);
	~FilterPipeline();

	/**
		Class Primary Usage Functions
	*/
	cv::Mat filter_image(const cv::Mat& src, bool show = false);

	// Class Modifying Functions
	int add_colorspace(ColorSpace cmap);
	int add_colorspace_limits(int8_t limits[6]);

	// Set Functions
	void set_colorspace_limits(int index, int8_t limits[6]);

	// Get Functions
	vector<int8_t> get_colorspace_limits();

	// Debugging Functions
	void print_internals();
};


#endif // FILTER_PIPELINE_H_
