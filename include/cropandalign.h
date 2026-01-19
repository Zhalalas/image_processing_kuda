#pragma once
#include <opencv2/core.hpp>
#include <tuple>
#include <string>
using namespace std;

tuple<cv::Mat, cv::Mat, cv::Mat> cropAndAlignByMat(
    const cv::Mat& image,
    const cv::Mat& det,
    const cv::Mat& landmarks,
    float template_scale = -1.0f,
    const string& template_mode = "default",
    int image_size = 112,
    bool allow_upscale = false
);