#pragma once
#include <opencv2/core.hpp>
#include <tuple>
#include <string>
using namespace std;

tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> alignFace(
    const cv::Mat& image,
    const cv::Mat& bbox,
    cv::Mat& original_landmarks,
    const string& template_mode = "default"
);