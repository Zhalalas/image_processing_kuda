#include <iostream>
#include <sstream>
#include <string>
#include <optional>
#include <tuple>
#include <utility>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <array>
#include <map>
#include "const.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include "alignface.h"
#include "writeimage.h"

using nlohmann::json;
using namespace std;


int main(){
    // ? Change the sample image name to test with different images from the samples folder
    string sample_image_name = "istockphoto-1344963248-612x612.jpg";
    string sample_image_ext = sample_image_name.substr(sample_image_name.find_last_of('.'));
    string sample_image_path = string(SAMPLES_PATH) + sample_image_name;

    // ? The details are automatically generated JSON files containing detection boxes and landmarks using our detection model
    filesystem::path details_path = filesystem::path(SAMPLES_PATH) / filesystem::path(sample_image_name);
    details_path.replace_extension(".txt");
    string sample_image_details_path = details_path.string();

    cout << "Sample image path: " << sample_image_path << endl;
    cout << "Sample image details path: " << sample_image_details_path << endl;

    cv::Mat sample_image = cv::imread(sample_image_path);
    if (sample_image.empty()) {
        cerr << "Failed to read image: " << sample_image_path << endl;
        return -1;
    }

    ifstream details_file(sample_image_details_path);
    if (!details_file.is_open()) {
        cerr << "Failed to open details file: " << sample_image_details_path << endl;
        return -1;
    }
    json sample_image_details;
    details_file >> sample_image_details;

    auto detections = sample_image_details["detections"];
    vector<float> bounding_box_vec = detections[0]["box"].get<vector<float>>();
    vector<float> landmarks_vec = detections[0]["landmarks"].get<vector<float>>();

    cv::Mat bounding_box(1, 4, CV_32F);
    cv::Mat landmarks(5, 2, CV_32F);


    auto [_, aligned_image, __, aligned_landmarks] =
        alignFace(sample_image, bounding_box, landmarks, "default");

    writeAlignedImage(SAMPLES_PATH, sample_image_name, aligned_image, aligned_landmarks);

    cout << "Sample image has been aligned successfully." << endl;
    return 0;
}
