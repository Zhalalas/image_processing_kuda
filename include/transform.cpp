#include "transform.h"
#include <opencv2/core.hpp>

cv::Mat transformLandmarks(
    const cv::Mat& landmarks, 
    const cv::Mat& transformation_matrix
){
    /*
    Apply a transformation matrix to landmarks.

    Args:
        landmarks: Array of landmarks to transform
        transformation_matrix: 2x3 transformation matrix

    Returns:
        Transformed landmarks
    */
    
    // Add homogeneous coordinate to landmarks
    cv::Mat ones = cv::Mat::ones(landmarks.rows, 1, landmarks.type());
    cv::Mat homogeneous_landmarks;
    cv::hconcat(landmarks, ones, homogeneous_landmarks);  

    // Apply transformation
    cv::Mat transformed = homogeneous_landmarks * transformation_matrix.t(); 
    homogeneous_landmarks.convertTo(homogeneous_landmarks, CV_32F);
    transformed = homogeneous_landmarks * transformation_matrix.t();
    return transformed; 
}
