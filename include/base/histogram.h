#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

#include "base/definitions.h"

using namespace std;

typedef enum HistogramAxis{
	AXIS_BOTTOM_VIEW  	= 0,
	AXIS_SIDE_VIEW   	= 1,
}HistogramAxis;


class histogram_t{
private:
	// Geometric Parameters
	int32_t _height;
	int32_t _width;

	// Statistics Variables
	vector<float> _averages;
	vector<int32_t> _values;

	// Misc Variables
	int32_t window_size;
public:
	histogram_t();
	~histogram_t();

	// Primary usage functions
	void create(cv::Mat image);
	void smooth(int32_t smoothing_window_size);
	void normalize();

	// Set functions
	void set_window_size(int32_t size);
	// Get functions
	vector<int32_t> get_average_values();
	vector<float> get_raw_values();

	// Debugging Functions
	void display();
};


#endif // HISTOGRAM_H_
