#pragma once
#include <opencv2/core.hpp>
using namespace std;

cv::Mat transformLandmarks(
    const cv::Mat& landmarks, 
    const cv::Mat& transformation_matrix
);