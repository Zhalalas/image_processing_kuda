#pragma once           
#include <opencv2/opencv.hpp>   
using namespace std;

cv::Mat resizeWithPadding(
    const cv::Mat& image,
    cv::Size target_size = cv::Size(112, 112)
);
