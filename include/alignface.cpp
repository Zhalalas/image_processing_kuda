#include "alignface.h"
#include "cropandalign.h"  // cropAndAlignByMat
#include "resizer.h"     // resizeWithPadding
#include <stdexcept>
using namespace std;

tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> alignFace(
    const cv::Mat& image,
    const cv::Mat& bbox,
    cv::Mat& original_landmarks,
    const string& template_mode = "default"
){
    /*
    Align a face to both 112x112 and 250x250 sizes while maintaining landmark consistency.

    Args:
        image: Original image containing the face
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        original_landmarks: Landmarks relative to original image (5, 2)

    Returns:
        AlignedFace object containing both sized images and their landmarks

    Raises:
        AppError: If alignment fails
    */
    // Validate inputs
    if (image.dims != 2 or image.channels() != 3){
        throw runtime_error("Input image must be 3-channel BGR");
    }

    if (bbox.total() != 4){
        throw runtime_error("Bounding box must have shape (4,)");
    }

    if (original_landmarks.rows != 5 || original_landmarks.cols != 2) {
    throw runtime_error("Landmarks must have shape (5, 2)");
    }
    // Align once to 250x250 (larger size for better quality)
    tuple<cv::Mat, cv::Mat, cv::Mat> aligned_result_250 = cropAndAlignByMat(
    image,
    bbox,
    original_landmarks,
    -1.0f,
    template_mode,
    250,
    true
    );

    cv::Mat aligned_250 = std::get<0>(aligned_result_250);
    cv::Mat ignored = std::get<1>(aligned_result_250); // you can ignore this if unused
    cv::Mat landmarks_250 = std::get<2>(aligned_result_250);
    cv::Mat aligned_112 = resizeWithPadding(aligned_250, cv::Size(112, 112));
    cv::Mat landmarks_112;
    landmarks_250.convertTo(landmarks_112, CV_32F); 
    landmarks_112 = landmarks_112 * (112.0 / 250.0);
    return make_tuple(aligned_112, aligned_250, landmarks_112, landmarks_250);
}
