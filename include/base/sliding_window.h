#ifndef SLIDING_WINDOW_H_
#define SLIDING_WINDOW_H_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

#include "base/definitions.h"

using namespace std;

class sliding_window_t{
private:
	// Geometric Parameters
	pixel_t _center;
	int32_t _height;
	int32_t _width;

	// Variables used to define how window moves incrementally
	int32_t _min_height;
	int32_t _max_height;
	int32_t _min_width;
	int32_t _max_width;

	// Processing Thresholds
	int32_t _pixel_thresh;

	// Misc Variables
	vector<pixel_t> _pixels;
public:
	sliding_window_t();
	~sliding_window_t();

	void init(pixel_t start_location);

	// Sliding Actions
	/**       Function: Sets the PWM channel dependant on steps on/off
     *    @param channel: Desired PWM channel to manipulate (0 - 15) -> Use -1 for all channels
     *    @param on_val:  Desired step value associated with the time on (0 - 4095)
     *    @param off_val: Desired step value associated with the time off (0 - 4095)
     */
	void move_right(bool flip);
	void move_left(bool flip);
	void move_up(bool flip);
	void move_down(bool flip);

	// Set Functions
	void set_limits(int32_t limits[4]);
	void set_threshold(int32_t threshold);
	void set_directions(int32_t up, int32_t right);
	void set_center(pixel_t location);
	// Get Functions
	vector<pixel_t> get_edges();
	int32_t get_pixel_count(cv::Mat image);

	// Debugging Functions
	void display(cv::Mat image);
};

#endif // SLIDING_WINDOW_H_
