#ifndef RANSAC_H_
#define RANSAC_H_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <algorithm>    // std::random_shuffle
#include <stdlib.h>      // std::rand, std::srand

typedef cv::Point_<uint16_t> pixel_t;

using namespace std;

class RansacLineFitter{
private:
	float _threshold;
	int _max_iterations = 100;
	int _iter = 0;

	cv::Mat _image;
	cv::Mat _gray;
	cv::Mat _nonzeros;
	cv::Mat _pcl_pts;
	std::vector<int> _ptsIn;
	vector<pixel_t> _pts;

	float _m = 1.0;
	float _intercept = 0.0;
	float _best_score = 0.0;
	float _a = _m;
	float _b = -1.0;
	float _c = _intercept;


	void loop_through_image(pixel_t &px);
	float computeDistance2Line(pixel_t &px);
	// // Parallel execution with function object.
	// struct PixelLooper{
	// 	void operator ()(pixel_t &px, const int * position) const{
	// 		// Perform a simple threshold operation
	// 		this->loop_through_image(pixel);
	// 	}
	// };
	int _randomize(int max);

public:
	RansacLineFitter();
	RansacLineFitter(double threshold, int iterations);
	~RansacLineFitter();

	void grab_image(const cv::Mat& image, bool show=true,bool debug_performance=true);

	void find_line();
	void update();

	void set_threshold(float threshold);
	void set_max_iterations(int iterations);

	float get_threshold();
	int get_max_iterations();

};

#endif // RANSAC_H_
