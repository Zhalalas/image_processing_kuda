#pragma once
#include <opencv2/core.hpp> 
using namespace cv;

void validateInputs(
    const cv::Mat& image,
    const cv::Mat& det,
    const cv::Mat& landmarks,
    float* template_scale = nullptr
);