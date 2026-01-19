#include "scale.h"
#include <opencv2/core.hpp>   
using namespace std;

cv::Mat scaleArcfaceTemplate(
    const cv::Mat& templ, double scale = 0.9, pair<int,int> center = {56, 56}
){
    /*
        Scale landmark template around the center to zoom out or in.

    Args:
        template: (N, 2) ndarray of ArcFace landmarks
        scale: Zoom factor. <1 = zoom out (face smaller)
        center: Center point to scale around (default: center of 112x112 image)

    Returns:
        New scaled template
    */
    cv::Mat t;
    templ.convertTo(t, CV_32F);
    int cx = center.first;
    int cy = center.second;
    cv::Mat center_mat = (cv::Mat_<float>(1, 2) << (float)cx, (float)cy);
    cv::Mat center_repeat;
    cv::repeat(center_mat, t.rows, 1, center_repeat);
    cv::Mat scaled;
    cv::subtract(t, center_repeat, scaled);
    scaled = scaled * static_cast<float>(scale);
    cv::add(scaled, center_repeat, scaled);
    return scaled;
}