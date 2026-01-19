#include "cropandalign.h"
#include "align.h"        // for alignUsingAllLandmarks
#include "validation.h"     // for validateInputs
#include <algorithm>
#include <stdexcept>
using namespace std;

tuple<cv::Mat, cv::Mat, cv::Mat> cropAndAlignByMat(
    const cv::Mat& image,
    const cv::Mat& det,
    const cv::Mat& landmarks,
    float template_scale = -1.0f,
    const string& template_mode = "default",
    int image_size = 112,
    bool allow_upscale = false
){
    /*
    Crop and align a facial image using detected landmarks.

    Args:
        image: Input image as numpy array or torch tensor (H, W, C)
        det: Detection bounding box as [x_min, y_min, x_max, y_max]
        landmarks: Facial landmarks coordinates
        template_scale: Optional scale factor for the template (0.8-1.2 recommended)
                        Values < 1 make the face appear smaller in the output
                        Values > 1 make the face appear larger in the output
        template_mode: Template mode to use ('arcface' or 'default')
        image_size: Output image size (default: 112)
        allow_upscale: Whether to allow upscaling images (default: False)
                       If False, the output size will be adjusted to prevent upscaling

    Returns:
        Tuple containing:
            - Cropped and aligned facial image ( numpy array )
            - Detection bounding box
            - Transformed facial landmarks in the aligned image

    Raises:
        ValueError: If the alignment method is invalid or image format is incorrect
        RuntimeError: If alignment fails
    */
    // Check if upscaling would occur and adjust image_size if not allowed
    if (!allow_upscale){
        // Estimate the face size from the detection box
        int face_width = det.at<int>(2) - det.at<int>(0);
        int face_height = det.at<int>(3) - det.at<int>(1);
        int face_size = max(face_width, face_height);

        // If requested size is larger than face size, adjust to prevent upscaling
        if (image_size > face_size){
            // Round down to nearest multiple of 8 for better compatibility
            int adjusted_size = face_size / 8 * 8;
            // Ensure minimum size of 32
            image_size = max(32, adjusted_size);
        }
    }
    // Validate inputs
    float scale = 1.0f;
    validateInputs(image, det, landmarks, &scale);

    int x_min = static_cast<int>(det.at<float>(0));
    int y_min = static_cast<int>(det.at<float>(1));
    int x_max = static_cast<int>(det.at<float>(2));
    int y_max = static_cast<int>(det.at<float>(3));

    if (x_max <= x_min || y_max <= y_min) {
        throw runtime_error("Invalid detection bbox");
    }
    int face_width = x_max - x_min;
    int face_height = y_max - y_min;
    int face_size = max(face_width, face_height);

    if (image_size > face_size) {
        image_size = max(32, (face_size / 8) * 8);  // ensure <= face size
    }
    if (landmarks.empty() || landmarks.rows != 5 || landmarks.cols < 2) {
        throw std::runtime_error("Invalid landmarks shape");
    }
    auto bbox = make_tuple(x_min, y_min, x_max, y_max);
    tuple<cv::Mat, cv::Mat, cv::Mat> aligned_result = alignUsingAllLandmarks(
        image,
        landmarks,
        bbox,
        -1.0f,
        template_mode,
        image_size
    );

    cv::Mat result_img = get<0>(aligned_result);
    cv::Mat result_det = get<1>(aligned_result);
    cv::Mat result_landmarks = get<2>(aligned_result);
    return make_tuple(result_img, result_det, result_landmarks);
}
