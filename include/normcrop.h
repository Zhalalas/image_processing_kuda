#pragma once
#include <opencv2/core.hpp>    // For cv::Mat
#include <opencv2/imgproc.hpp> // For cv::warpAffine
#include <string>
#include <tuple>
#include <optional>
using namespace std;

cv::Mat normCrop(
    const cv::Mat& image,
    const cv::Mat& landmark,
    int image_size,
    const string mode,
    float template_scale = -1.0f
);