#include "resizer.h"
#include <opencv2/opencv.hpp>
#include <algorithm> 
using namespace std;


cv::Mat resizeWithPadding(
    const cv::Mat& image,
    cv::Size target_size = cv::Size(112, 112)
){
    /*
    Resize an image with padding to maintain aspect ratio.

    Args:
        image: The image to resize
        target_size: The target size (width, height)

    Returns:
        The resized image
    */
    int height = image.rows;
    int width = image.cols;
    int max_dim = max(height, width);

    //Calculate the padding needed to make the image square using max_dim
    int pad_top = (max_dim - height) / 2;
    int pad_bottom = max_dim - height - pad_top;
    int pad_left = (max_dim - width) / 2;
    int pad_right = max_dim - width - pad_left;

    //Pad the image with zeros
    cv::Mat padded_image;
    cv::copyMakeBorder(
        image, padded_image, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0)
    );

    //Resize the padded image to the target size
    cv::Mat resized_image;
    cv::resize(padded_image, resized_image, target_size, 0, 0, cv::INTER_LINEAR);
    return resized_image; 
}