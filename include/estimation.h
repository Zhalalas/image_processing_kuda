#pragma once

#include <opencv2/core.hpp>      // For cv::Mat
#include <optional>
#include <utility>
#include <string>
using namespace std;

pair<cv::Mat, int> estimateNorm(
    const cv::Mat& landmark,
    int image_size,
    const string& mode,
    optional<float> template_scale
);