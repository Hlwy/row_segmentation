#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

#include "filters/color_filter.h"
#include "filters/filter_pipeline.h"

using namespace std;
using namespace arma;

int h = 480; int w = 640;
ColorSpace spaces[] = {YUV, HSV};

vector<uint8_t> hsv_ll = {32, 52, 0};		// Alright Case
vector<uint8_t> hsv_ul = {107, 255, 255};	// Alright Case
// int hsv_ll[] = {18, 31, 25};	// Test Case
// int hsv_ul[] = {111, 255, 130}; // Test Case


vector<uint8_t> yuv_ll = {0, 0, 0};		// Original Case
vector<uint8_t> yuv_ul = {164, 126, 126};	// Original Case
// int yuv_ll[] = {72, 107, 107};	// Test Case
// int yuv_ul[] = {148, 129, 135};	// Test Case


/**  @function main */
int main(int argc, char** argv){
	umat lims;
	lims << 0 << 0 << 0 << 164 << 126 << 126 << endr
		<< 32 << 52 << 0 << 107 << 255 << 255 << endr;

	// OS Variables
	char* default_img_path = "/home/hunter/data/training_raw/early_season/1/frames/frame1.jpg";
	char* img_path;
	// Labels
	string source_window = "Source Image";

	// Image Container Initializations
	cv::Mat src;		// Raw, Unknown Sized, Source Image (BGR Color Space)
	cv::Mat ssrc; 		// Raw Source Image resized to defined size standard
	cv::Mat cfilled; 	// Image resulting from combined color filter pre-processing

	// Command-line Parsing
	if(argc < 2){
		cout << "Usage: " << argv[0] << " <path_to_image>\r\n"
		<< "	Using default image path " << default_img_path << endl;
		img_path = default_img_path;
	}else img_path = argv[1];

	// Object Declarations
	FilterPipeline pipe(spaces,lims);

	// Load an image
	src = cv::imread(img_path,1);

	// Double check if source image stored properly and if so resize to standard
	if(src.empty()){
		cerr << "Loading Source image failed. (No loaded data)" << endl;
		return -1;
	}else{
		cv::resize(src,ssrc,cv::Size(w,h));
	}

	pipe.filter_image(ssrc,true);


	// Show Image
	// cv::namedWindow(source_window, CV_WINDOW_NORMAL);
	// cv::imshow(source_window, src);
	// Wait until user exits the program
	cv::waitKey(0);

	return 0;
}
