#ifndef FILTER_PIPELINE_H_
#define FILTER_PIPELINE_H_

#include <armadillo>
#include "color_filter.h"

using namespace std;
using namespace arma;

typedef enum MorphType{
	OPEN      = 0,
	CLOSE     = 1,
	ERODE     = 2,
	DILATE 	= 3,
	num_types
}MorphType;

typedef enum MorphElement{
	RECTANGLE	= cv::MORPH_RECT,
	ELLIPSE	= cv::MORPH_ELLIPSE,
	num_elements
}MorphElement;

class FilterPipeline{
private:
	string _id;		// Class string Identifier
	int _ccount = 0;	// Num of Filters Used
	ucube _limits; 	// Upper and Lower Thresholding Limits per channel

	// Filter-specific containers
	vector<string> _lbls;		// Color Space identifiers for debugging
	vector<ColorSpace> _cmaps;	// Color space that color filter is operating in

	cv::Mat _filtered;			// Color Filtered Image of source image
	cv::Mat _mask;				// Mask resultanting from color filtering
public:
	vector<ColorFilter*> filters; // ColorFilter Objects
	/**
		Class Constructors/Deconstructors and Overloads
	*/
	FilterPipeline();
	FilterPipeline(ColorSpace cmaps[],umat limits, bool verbose = false);
	~FilterPipeline();

	/**
		Class Primary Usage Functions
	*/
	cv::Mat filter_image(const cv::Mat& src, bool show = false);
	cv::Mat blur_filtered(const cv::Mat& src,int aperture = 7,bool show = false);
	cv::Mat morph_filtered(const cv::Mat& src, uint8_t kernel_size[2], MorphElement shape, MorphType morphing,bool show = false);

	// Class Modifying Functions
	void add_color_filter(ColorSpace cmap, umat limits, bool verbose = false);

	// Set Functions
	void set_color_filter_limits(int index, umat limits, bool verbose = false);

	// Get Functions
	ucube get_colorspace_limits();
	int get_colorspace_count();

	// Debugging Functions
	void print_internals();
};


#endif // FILTER_PIPELINE_H_
