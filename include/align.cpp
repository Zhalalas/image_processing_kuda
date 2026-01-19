#include "align.h"
#include "normcrop.h"       // for normCrop
#include "estimation.h"      // for estimateNorm
#include "transform.h"       // for transformLandmarks
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <sstream>
using namespace std;

tuple<cv::Mat, cv::Mat, cv::Mat> alignUsingAllLandmarks(
    const cv::Mat& image,
    const cv::Mat& landmark,
    const tuple<int, int, int, int>& bbox,
    float template_scale = -1.0f,
    const string& template_mode = "default",
    int image_size = 112
){
    /*
    Align face using all 5 facial landmarks.

    Args:
        image: Input image
        landmarks: Facial landmarks (5 points)
        bbox: Bounding box as (x_min, y_min, x_max, y_max)
        template_scale: Optional scale factor for the template
        template_mode: Template mode to use ('arcface' or 'default')
        image_size: Output image size

    Returns:
        Tuple of (aligned_image, detection, transformed_landmarks)
    */
    try{
        // Get transformation matrix
        pair<cv::Mat, int> norm_result = estimateNorm(landmark, image_size, template_mode, template_scale);
        cv::Mat M = norm_result.first;
        int pose_index = norm_result.second;

        // Use similarity transform to align the face
        cv::Mat cropped_aligned_img = normCrop(image, landmark, image_size, template_mode, template_scale);
        // Transform landmarks using the same transformation matrix
        if (M.empty()) {
        throw runtime_error("Transformation matrix is empty. Cannot transform landmarks.");
        }

        cv::Mat landmark_f;
        landmark.convertTo(landmark_f, CV_32F);  // ensure float type
        cv::Mat transformed_landmarks = transformLandmarks(landmark_f, M);

        int bbox_x1 = get<0>(bbox);
        int bbox_y1 = get<1>(bbox);
        int bbox_x2 = get<2>(bbox);
        int bbox_y2 = get<3>(bbox);
        cv::Mat bbox_mat = (cv::Mat_<int>(1, 4) << bbox_x1, bbox_y1, bbox_x2, bbox_y2);
        return make_tuple(cropped_aligned_img, bbox_mat, transformed_landmarks);
    }
    catch (const exception& e){
        cerr << "Original error: " << e.what() << endl;  // <- inspect in console/log
        ostringstream oss;
        oss << "All-landmarks alignment failed: " << e.what();
        throw runtime_error(oss.str());
    }
}
