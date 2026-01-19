#include "writeimage.h"
#include <opencv2/imgproc.hpp>  // for cv::circle
#include <opencv2/imgcodecs.hpp> // for cv::imwrite
#include <iostream>
#include <filesystem>
#include <cmath>
using namespace std;

void writeAlignedImage(
    const string& write_path,
    const string& write_name,
    const cv::Mat& aligned_image,
    const cv::Mat& aligned_landmarks
){
    size_t pos = write_name.find_last_of('.');
    string base = write_name.substr(0, pos);
    string write_ext = write_name.substr(pos + 1);

    string write_new_name = base + "_aligned." + write_ext;

    filesystem::path folder_path(write_path);
    filesystem::path full_path = folder_path / write_new_name;

    // ? Draw landmarks and bbox on the aligned image for visualization
    cv::Mat vis_image = aligned_image.clone();
    for (int i = 0; i < aligned_landmarks.rows; ++i) {
    int x = static_cast<int>(std::round(aligned_landmarks.at<float>(i, 0)));
    int y = static_cast<int>(std::round(aligned_landmarks.at<float>(i, 1)));

    cv::circle(vis_image, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
    }
    // Ensure we write to the constructed full path (folder + new filename)
    cv::imwrite(full_path.string(), vis_image);
    cout << "Aligned image saved to: " << full_path.string() << endl;
}
