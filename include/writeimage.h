#pragma once
#include <string>
#include <opencv2/core.hpp>
using namespace std;

void writeAlignedImage(
    const string& write_path,
    const string& write_name,
    const cv::Mat& aligned_image,
    const cv::Mat& aligned_landmarks
);