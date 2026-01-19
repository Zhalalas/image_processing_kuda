#pragma once
#include <opencv2/core.hpp>     // cv::Mat
#include <tuple>
#include <string>
using namespace std;

tuple<cv::Mat, cv::Mat, cv::Mat> alignUsingAllLandmarks(
    const cv::Mat& image,
    const cv::Mat& landmark,
    const tuple<int, int, int, int>& bbox,
    float template_scale = -1.0f,
    const string& template_mode = "default",
    int image_size = 112
);