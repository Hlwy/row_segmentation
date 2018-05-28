#include "algorithms/ransaclinefitter.h"
#include <time.h>
#include <random>
#include <iostream>
using namespace std;

RansacLineFitter::RansacLineFitter(){

	// m_DistDenominator = sqrt(m_a * m_a + m_b * m_b); // Cache square root for efficiency
}

RansacLineFitter::RansacLineFitter(double threshold, int iterations){
	this->_threshold = threshold;
	this->_max_iterations = iterations;
}


RansacLineFitter::~RansacLineFitter(){

}

void RansacLineFitter::grab_image(const cv::Mat& image, bool show, bool debug_performance){
	int start = 0, end = 0;

	if(debug_performance) start = cv::getTickCount();

	cv::Mat tmp, gray, nonzero;
	tmp = image.clone();
	cout << "[grab_image] --- Input size: " << tmp.rows << ", " << tmp.cols << ", " << tmp.channels() << ", " << tmp.total() << endl;
	this->_image = tmp;

	cv::cvtColor(tmp, gray, CV_BGR2GRAY);
	cv::threshold(gray, gray, 5, 255, cv::THRESH_BINARY);
	this->_gray = gray.clone();
	cv::findNonZero(gray,nonzero);
	cout << "[grab_image] --- nonzero size: " << nonzero.rows << ", " << nonzero.cols << ", " << nonzero.size << endl;
	this->_nonzeros = nonzero;

	if(debug_performance){
		end = cv::getTickCount();
		float dt = float(end - start) / cv::getTickFrequency();
		cout << "[grab_image] Time taken to finish: " << dt * 1000.0 <<  " [ms], " << (1.0 / dt) << " FPS" << endl;
	}

	if(show){
		string lbl = "RansacLineFitter: Nonzero Image";
		cv::namedWindow(lbl, CV_WINDOW_NORMAL);
		cv::imshow(lbl,gray);
	}
}

void RansacLineFitter::loop_through_image(pixel_t &px){
	if(this->_gray.at<uchar>(px.y, px.x) > 0){
		// cout << i << ", " << j << endl;     // Do your operations
	}
}


void RansacLineFitter::update(){
	cv::Mat tmpImg = this->_gray;
	int n_samples = tmpImg.total();
	int h = tmpImg.rows, w = tmpImg.cols;
	int start = cv::getTickCount();
	int iter = 0;
	int n_valid = 0, n_bad = 0;
	int prev_idx = this->_randomize(n_samples);
	int idx = 0;
	while(iter <= this->_max_iterations){
		// Randomly select 2 sample points
		if(idx == prev_idx){	// Grab New sample index if the same as before
			idx = this->_randomize(n_samples);
			cout << "New Pixel index is the same as the previous iteration --- " << idx << endl;
			continue;
		}
		// Grab the new pixel location at that index
		if(tmpImg.at<uchar>(idx) > 0){
			// inlierGuess = 0;
			n_valid++;
			cout << "Found a Valid pixel, Continueing loop --- " << n_valid << endl;
		}else{
			n_bad++;
			cout << "Not a Valid pixel, Continueing loop -- " << n_bad << endl;
			continue;
		}
	// 	m_m = (Point2->m_Point2D[1] - Point1->m_Point2D[1]) / (Point2->m_Point2D[0] - Point1->m_Point2D[0]); // Slope
	// m_d = Point1->m_Point2D[1] - m_m * Point1->m_Point2D[0]; // Intercept
		// // Compute the distances between all points with the fitting line
		// kLine = sample(:,2)-sample(:,1); // two points relative distance
		// kLineNorm = kLine/norm(kLine);
		// normVector = [-kLineNorm(2),kLineNorm(1)]; // Ax+By+C=0 A=-kLineNorm(2),B=kLineNorm(1)
		// distance = normVector*(data - repmat(sample(:,1),1,number));
		// // Compute the inliers with distances smaller than the threshold
		// inlierIdx = find(abs(distance)<=threshDist);
		// inlierNum = length(inlierIdx);
		// // Update the number of inliers and fitting model if better model is found
		// if inlierNum>=round(inlierRatio*number) && inlierNum>bestInNum{
		// 	bestInNum = inlierNum;
		// 	parameter1 = (sample(2,2)-sample(2,1))/(sample(1,2)-sample(1,1));
		// 	parameter2 = sample(2,1)-parameter1*sample(1,1);
		// 	bestParameter1=parameter1; bestParameter2=parameter2;
		// }

		idx = prev_idx;
		iter++;
	}

	int end = cv::getTickCount();
	// std::cout << "RANSAC took: " << GRANSAC::VPFloat(end - start) / GRANSAC::VPFloat(cv::getTickFrequency()) * 1000.0 << " ms." << std::endl;
	cout << "Good, Bad: " << n_valid << ", " << n_bad << endl;
}

float RansacLineFitter::computeDistance2Line(pixel_t &px){
	// float Numer = fabs(this->_a * px.x + this->_intercept * ExtPoint2D->m_Point2D[1] + m_c);
	// float Dist = Numer / m_DistDenominator;

	// // Debug
	// std::cout << "Point: " << ExtPoint2D->m_Point2D[0] << ", " << ExtPoint2D->m_Point2D[1] << std::endl;
	// std::cout << "Line: " << m_a << " x + " << m_b << " y + "  << m_c << std::endl;
	// std::cout << "Distance: " << Dist << std::endl << std::endl;
}







void RansacLineFitter::find_line(){
	int start = cv::getTickCount();
	// this->_estimator.Estimate()
	int end = cv::getTickCount();

	// std::vector<int> inliers;
	// pcl::PointCloud<pcl::PointXYZRGB>::Ptr final (new pcl::PointCloud<pcl::PointXYZRGB>);
	//
	// // created RandomSampleConsensus object and compute the appropriated model
	// SampleConsensusModelLine<PointXYZRGB>::Ptr _model(new SampleConsensusModelLine<PointXYZRGB> (this->_cloud));
	// RandomSampleConsensus<PointXYZRGB> ransac(_model);
	// ransac.setDistanceThreshold(.01);
	// ransac.computeModel();
	// ransac.getInliers(inliers);
	//
	//
	// // copies all inliers of the model computed to another PointCloud
	// copyPointCloud<PointXYZRGB>(*this->_cloud, inliers, *final);
	// this->draw_cloud("Ransac",this->_cloud);

	// Find the best model using RANSAC

	// double loss = this->_ransac->FindBest(this->_model, this->data, data.size(), 2);
	//
	// // Determine inliers using the best model if necessary
	// vector<int> inliers = ransac.FindInliers(model, data, data.size());

	// Print the result
	// cout << "- True Model:  " << trueModel << endl;
	// cout << "- Found Model: " << model << " (Loss: " << loss << ")" << endl;
	// cout << "- The Number of Inliers: " << inliers.size() << " (N: " << data.size() << ")" << endl;

}





void RansacLineFitter::set_threshold(float threshold){this->_threshold = threshold;}
void RansacLineFitter::set_max_iterations(int iterations){this->_max_iterations = iterations;}

float RansacLineFitter::get_threshold(){return this->_threshold;}
int RansacLineFitter::get_max_iterations(){return this->_max_iterations;}


int RansacLineFitter::_randomize(int max){
	srand((unsigned)time(0));
	int min = 0; int range = (max - min);
	int rnd = min + int((range * rand()) / (RAND_MAX + 1.0));
	return rnd;
}
