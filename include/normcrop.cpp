#include "normcrop.h"
#include "estimation.h"   // for estimateNorm
#include <opencv2/imgproc.hpp>  // For warpAffine
#include <opencv2/core.hpp>
using namespace std;

cv::Mat normCrop(
    const cv::Mat& image,
    const cv::Mat& landmark,
    int image_size,
    const string mode,
    float template_scale = -1.0f
){
    /*
    Normalize and crop the face image using facial landmarks.

    Args:
        img: Input image
        landmark: 5-point facial landmarks
        image_size: Output image size (default: 112)
        mode: Template mode ('arcface' or 'default')
        template_scale: Optional scale factor for the template

    Returns:
        Normalized and cropped face image
    */
    auto[M, posed_index] = estimateNorm(landmark, image_size, mode, template_scale);
    cv::Mat M2x3 = M(cv::Rect(0, 0, 3, 2));
    cv::Mat warped;
    cv::warpAffine(image, warped, M2x3, cv::Size(image_size, image_size), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return warped;
}
