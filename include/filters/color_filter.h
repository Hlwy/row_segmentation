#ifndef COLOR_FILTER_H_
#define COLOR_FILTER_H_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <stdint.h>

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
	ColorSpace _cmap;	// Color space that color filter is operating in
	string _lbl;		// Color Space identifier for debugging

	// Upper and Lower Thresholding Limits per channel
	vector<uint8_t> _upper_limits = {255, 255, 255};
	vector<uint8_t> _lower_limits = {0, 0, 0};
public:
	cv::Mat filtered;	// Color Filtered Image of source image
	cv::Mat mask;		// Mask resultanting from color filtering
	/**
		Class Constructors/Deconstructors and Overloads
	*/
	ColorFilter();
	ColorFilter(ColorSpace cmap);
	~ColorFilter();

	/**
		Class Primary Usage Functions
	*/
	void filter_color(const cv::Mat& src, bool show = false);
	cv::Mat blur_filtered(const cv::Mat& src,int aperture = 7,bool show = false);
	cv::Mat morphed_filtered(uint8_t kernel_size[2], bool use_opened = false,bool show = false);

	// Set Functions
	void set_colorspace(ColorSpace cmap);
	void set_upper_limits(vector<uint8_t> limits, bool verbose = false);
	void set_lower_limits(vector<uint8_t> limits, bool verbose = false);

	// Get Functions
	vector<uint8_t> get_upper_limits();
	vector<uint8_t> get_lower_limits();

	// Debugging Functions
	void print_internals();
	void print_internals(string label);
};


#endif // COLOR_FILTER_H_
