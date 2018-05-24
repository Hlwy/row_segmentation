#ifndef COLOR_FILTER_H_
#define COLOR_FILTER_H_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

using namespace std;

typedef enum ColorSpace{
	HSV       = 0,
	YUV       = 1,
	RGB     	= 2,
	BGR 		= 3,
	num_spaces
}ColorSpace;

class ColorFilter{
private:
	ColorSpace _cmap;			 // Color space that color filter is operating in
	string _space_desc;			 // Color Space identifier for debugging
	cv::Mat _img_space;			 // Copy of raw source image into specified color space
	cv::Mat _mask;				 // Mask resultanting from color filtering

	// Upper and Lower Thresholding Limits per channel
	vector<int32_t> _upper_limits = {255, 255, 255};
	vector<int32_t> _lower_limits = {0, 0, 0};
public:
	/**
		Class Constructors/Deconstructors and Overloads
	*/
	ColorFilter();
	ColorFilter(ColorSpace cmap);
	~ColorFilter();

	/**
		Class Primary Usage Functions
	*/
	cv::Mat filter_image(const cv::Mat& src, bool show = false);

	// Set Functions
	void set_colorspace(ColorSpace cmap);
	void set_upper_limits(int32_t limits[3]);
	void set_lower_limits(int32_t limits[3]);

	// Get Functions
	vector<int32_t> get_upper_limits();
	vector<int32_t> get_lower_limits();

	// Debugging Functions
	void print_internals();
};


#endif // COLOR_FILTER_H_