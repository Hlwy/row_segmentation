#ifndef LINE_H_
#define LINE_H_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

#include "base/definitions.h"

using namespace std;

class line_t{
private:
	// Line descriptor variables
	point_t _start;
	point_t _end;
	vector<point_t> _pts;
	// Line Equation Parameters: y = mx + b
	float _m;
	float _b;
	// Misc Descriptors
	float _length;
	float _angle;
	// Flags
	bool _is_detected;
public:
	line_t(point_t start_pt, point_t end_pt);
	~line_t();
};

#endif // LINE_H_
