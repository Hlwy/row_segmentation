#ifndef IMAGE_PROCESSING_UTILS_H_
#define IMAGE_PROCESSING_UTILS_H_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>

using namespace std;
/** PCL
	RANSAC: http://docs.pointclouds.org/trunk/group__sample__consensus.html
	http://www.jeffdelmerico.com/wp-content/uploads/2014/03/pcl_tutorial.pdf
	http://www.yanivresearch.info/writtenMaterial/RANSAC.pdf
	https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=5194&context=etd
*/

// OpenCV Best Practices: https://docs.opencv.org/2.4/modules/core/doc/intro.html#api-concepts
// OpenCV Tutorials: https://docs.opencv.org/3.0-beta/doc/tutorials/tutorials.html

/** OpenCV Image Processing References:
	https://docs.opencv.org/2.4/modules/imgproc/doc/imgproc.html
	https://docs.opencv.org/2.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html#smoothing
	https://www.theseus.fi/bitstream/handle/10024/140208/aalamaki_thesis.pdf?sequence=1&isAllowed=y
*/

// Array Operations: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#void%20inRange(InputArray%20src,%20InputArray%20lowerb,%20InputArray%20upperb,%20OutputArray%20dst)

// Contour Functions: https://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/moments/moments.html#moments

/** Histogram References:
	https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_equalization/histogram_equalization.html#histogram-equalization
	https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html#histogram-calculation
	https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html#template-matching
*/

// ColorSpaces Reference: https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/

/** Misc References:
	https://stackoverflow.com/questions/23468537/differences-of-using-const-cvmat-cvmat-cvmat-or-const-cvmat
	http://www.uio.no/studier/emner/matnat/its/UNIK4690/v18/labs/kompendium_maskinsyn.pdf
**/

#endif // IMAGE_PROCESSING_UTILS_H_
