#ifndef POINTCLOUD_UTILS_H_
#define POINTCLOUD_UTILS_H_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;
using namespace pcl;

// // PointCloud<PointXYZ>::Ptr _cloud;
// PointCloud<PointXYZRGB>::Ptr _cloud;
// PointCloud<PointXYZ>::Ptr _inliers;
// SampleConsensusModelLine<PointXYZ>::Ptr line_model;
// ModelCoefficients::Ptr _coefs;

PointCloud<PointXYZRGB>::Ptr mat2pclrgb(const cv::Mat& image, cv::Mat &coords);
PointCloud<PointXYZRGB>::Ptr mat2pcl(const cv::Mat& image, cv::Mat &coords);
cv::Mat pcl2matrgb(const PointCloud<PointXYZRGB>::Ptr &cloud_in, cv::Mat &coords, cv::Mat& image);
cv::Mat pcl2mat(const PointCloud<PointXYZRGB>::Ptr &cloud_in, cv::Mat &coords, cv::Mat& image);
void draw_cloud(const std::string &text, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);
void convertpcl();



PointCloud<PointXYZRGB>::Ptr RansacLineFitter::mat2pclrgb(const cv::Mat& image, cv::Mat &coords){
	int rows = image.rows;
	int cols = image.cols;

	PointCloud<PointXYZRGB>::Ptr cloud_ptr(new PointCloud<PointXYZRGB>);

	for(int row = 0; row < rows;row++){
		for(int col = 0; col < cols; col++){
			// std::cout << "X, Y: " << col << ", " << row << std::endl;

			PointXYZRGB point;
			point.x = coords.at<double>(0,row*cols+col);
			point.y = coords.at<double>(1,row*cols+col);
			point.z = coords.at<double>(2,row*cols+col);

			cv::Vec3b color = image.at<cv::Vec3b>(cv::Point(col,row));
			uint8_t r = (color[2]);
			uint8_t g = (color[1]);
			uint8_t b = (color[0]);

			int32_t rgb = (r << 16) | (g << 8) | b;
			point.rgb = *reinterpret_cast<float*>(&rgb);

			cloud_ptr->points.push_back(point);
		}
	}
	return cloud_ptr;
}

PointCloud<PointXYZRGB>::Ptr RansacLineFitter::mat2pcl(const cv::Mat& image, cv::Mat &coords){
	int rows = image.rows;
	int cols = image.cols;

	PointCloud<PointXYZRGB>::Ptr cloud_ptr(new PointCloud<PointXYZRGB>);

	for(int row = 0; row < rows;row++){
		for(int col = 0; col < cols; col++){
			if(image.at<uchar>(row, col) > 0){
				// std::cout << "X, Y: " << col << ", " << row << std::endl;

				PointXYZRGB point;
				point.x = coords.at<double>(0,row*cols+col);
				point.y = coords.at<double>(1,row*cols+col);
				point.z = coords.at<double>(2,row*cols+col);

				uint8_t r = (255);
				uint8_t g = (255);
				uint8_t b = (255);

				int32_t rgb = (r << 16) | (g << 8) | b;
				point.rgb = *reinterpret_cast<float*>(&rgb);

				cloud_ptr->points.push_back(point);
			}
		}
	}
	return cloud_ptr;
}


cv::Mat RansacLineFitter::pcl2matrgb(const PointCloud<PointXYZRGB>::Ptr &cloud_in,cv::Mat& coords, cv::Mat& image){
	int h = image.rows;
	int w = image.cols;
	cout << "Cloud Input [h,w]: " << h << ", " << w << endl;

	coords = cv::Mat(3, cloud_in->points.size(), CV_64FC1);
	image = cv::Mat(h,w, CV_8UC3);

	for(int row = 0; row < h;row++){
		for(int col = 0; col < w; col++){
			if(image.at<uchar>(row, col) > 0) {
				// PointXYZRGB point = cloud_in->at(col,row);
				coords.at<double>(0,row*w+col) = cloud_in->points.at(row*w+col).x;
				coords.at<double>(1,row*w+col) = cloud_in->points.at(row*w+col).y;
				coords.at<double>(2,row*w+col) = cloud_in->points.at(row*w+col).z;

				cv::Vec3b color = cv::Vec3b(cloud_in->points.at(row*image.cols+col).b,
									   cloud_in->points.at(row*image.cols+col).g,
									   cloud_in->points.at(row*image.cols+col).r);

				image.at<cv::Vec3b>(cv::Point(col,row)) = color;
			}
		}
	}
	return image;
}

cv::Mat RansacLineFitter::pcl2mat(const PointCloud<PointXYZRGB>::Ptr &cloud_in,cv::Mat& coords, cv::Mat& image){
	int h = image.rows;
	int w = image.cols;
	cout << "Cloud Input [h,w]: " << h << ", " << w << endl;

	coords = cv::Mat(3, cloud_in->points.size(), CV_64FC1);
	image = cv::Mat(h,w, CV_8UC3);

	for(int row = 0; row < h;row++){
		for(int col = 0; col < w; col++){
			// PointXYZRGB point = cloud_in->at(col,row);
			if(image.at<uchar>(row, col) > 0) {
				// cout << i << ", " << j << endl;     // Do your operations
				coords.at<double>(0,row*w+col) = cloud_in->points.at(row*w+col).x;
				coords.at<double>(1,row*w+col) = cloud_in->points.at(row*w+col).y;
				coords.at<double>(2,row*w+col) = cloud_in->points.at(row*w+col).z;

				cv::Vec3b color = cv::Vec3b(255, 255, 255);

				image.at<cv::Vec3b>(cv::Point(col,row)) = color;
			}
		}
	}
	return image;
}


void RansacLineFitter::convertpcl(){

	// cv::Mat coords(3, tmp.cols * tmp.rows, CV_64FC1);
	// for(int col = 0; col < coords.cols; ++col){
	// 	coords.at<double>(0, col) = col % tmp.cols;
	// 	coords.at<double>(1, col) = col / tmp.cols;
	// 	coords.at<double>(2, col) = 10;
	// }
	//
	// this->_pcl_pts = coords.clone();
	// this->_cloud = this->mat2pcl(gray,coords);
	// // cv::Mat converted = this->pcl2mat(this->_cloud,coords,tmp);
	// cv::Mat converted = this->pcl2mat(this->_cloud,coords,gray);

	// if(show){
	// 	string lbl = "RansacLineFitter: Converted Image";
	// 	string lblNon = "RansacLineFitter: Nonzero Image";
	// 	cv::namedWindow(lbl, CV_WINDOW_NORMAL);
	// 	cv::namedWindow(lblNon, CV_WINDOW_NORMAL);
	// 	// cv::imshow(lbl,converted);
	// 	cv::imshow(lblNon,gray);
	// }
}


void draw_cloud(const std::string &text, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud){
	pcl::visualization::CloudViewer viewer(text);
	viewer.showCloud(cloud);
	while (!viewer.wasStopped()){}
}

#endif // POINTCLOUD_UTILS_H_
