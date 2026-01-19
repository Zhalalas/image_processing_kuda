#include "validation.h"
#include <sstream>       // for std::ostringstream
#include <stdexcept>
using namespace std;

void validateInputs(
    const cv::Mat& image,
    const cv::Mat& det,
    const cv::Mat& landmarks,
    float* template_scale = nullptr
){
    /*
    Validate input parameters.

    Args:
        image: Input image
        det: Detection bounding box
        landmarks: Facial landmarks
        alignment_method: Alignment method
        template_scale: Optional scale factor for the template
        template_mode: Template mode to use

    Raises:
        ValueError: If any input is invalid
    */
    //Check image format
    if (image.channels() != 3 || image.dims != 2){
        ostringstream oss;
        oss << "Expected RGB image with shape (H, W, 3), got "
        << image.rows << "x" << image.cols << "x" << image.channels();
        throw std::runtime_error(oss.str());
    }

    //Check detection box format
    if (det.total() != 4){
        ostringstream oss;
        oss << "Expected detection box with 4 values, got" << det.total();
        throw runtime_error(oss.str());
    }
    //Check landmarks format (should be 5 landmarks with x,y coordinates)
    if (landmarks.total() != 10){  // 5 landmarks * 2 coordinates
        ostringstream oss;
        oss << "Expected 5 landmarks (10 values), got " << landmarks.total();
        throw runtime_error(oss.str());
        }

    //Check template scale if provided
    if (template_scale != nullptr && (*template_scale <= 0.0f || *template_scale > 2.0f)){
        ostringstream oss;
        oss << "Invalid template scale: " << *template_scale << "Value should be positive and preferably between 0.8 and 1.2.";
        throw runtime_error(oss.str());
    }
}