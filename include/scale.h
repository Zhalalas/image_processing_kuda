#pragma once

#include <opencv2/core.hpp>   // For cv::Mat
#include <utility>  
using namespace std;

cv::Mat scaleArcfaceTemplate(
    const cv::Mat& templ, double scale = 0.9, pair<int,int> center = {56, 56}
);