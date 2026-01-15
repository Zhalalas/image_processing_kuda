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

using nlohmann::json;
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

pair<cv::Mat, int> estimateNorm(
    const cv::Mat& landmark,
    int image_size,
    const string& mode,
    optional<float> template_scale
){
    /*
    Estimate the transformation matrix for face normalization.

    Args:
        landmark: 5-point facial landmarks
        image_size: Output image size
        mode: Template mode ('arcface' or 'default')
        template_scale: Optional scale factor for the template

    Returns:
        Tuple of (transformation_matrix, pose_index)
    */

    // Validate input landmarks
    if (landmark.rows != 5 || landmark.cols != 2){
        ostringstream oss;
        oss << "Expected 5x2 landmarks, got "
            << landmark.rows << "x" << landmark.cols;
        throw runtime_error(oss.str());
    }

    // Create transformation estimator
    cv::Mat tform;

    // Add homogeneous coordinate
    cv::Mat ones = cv::Mat::ones(5, 1, CV_32F);
    cv::Mat lmk_tran;
    landmark.convertTo(lmk_tran, CV_32F);   // ensure float type
    cv::hconcat(lmk_tran, ones, lmk_tran);  // shape: 5x3

    // Initialize variables to track best transformation
    cv::Mat min_M;
    int min_index = -1;
    double min_error = numeric_limits<double>::infinity();

    // Container for templates to evaluate
    vector<cv::Mat> templates;

    if (mode == "arcface"){
        // Try to get pre-calculated template from map (convert Landmark5 -> cv::Mat)
        cv::Mat src;
        auto it = ARC_FACE_TEMPLATE_MAP.find(image_size);

        if (it == ARC_FACE_TEMPLATE_MAP.end()){
            // If not found in map, convert 112 template and scale it dynamically
            Landmark5 base_landmark = ARC_FACE_TEMPLATE_MAP.at(112);
            cv::Mat base_src(5, 2, CV_32F);
            for (int r = 0; r < 5; ++r){
                base_src.at<float>(r, 0) = base_landmark[r][0];
                base_src.at<float>(r, 1) = base_landmark[r][1];
            }
            double scale_factor = image_size / 112.0;
            cv::Point center(56, 56);  // Center of the 112x112 template
            src = scaleArcfaceTemplate(base_src, scale_factor, {center.x, center.y});

            // Adjust center point for the new image size
            // Create 1x2 matrices for old and new centers
            cv::Mat old_center_32f = (cv::Mat_<float>(1,2) << 56.f, 56.f);
            cv::Mat new_center_32f = (cv::Mat_<float>(1,2) << image_size / 2.0f, image_size / 2.0f);
            old_center_32f.convertTo(old_center_32f, src.type());
            new_center_32f.convertTo(new_center_32f, src.type());
            src = src - cv::repeat(old_center_32f, src.rows, 1) + cv::repeat(new_center_32f, src.rows, 1);

        }
        else{
            // Convert stored Landmark5 into cv::Mat
            Landmark5 lm = it->second;
            cv::Mat tmp(5, 2, CV_32F);
            for (int r = 0; r < 5; ++r){
                tmp.at<float>(r, 0) = lm[r][0];
                tmp.at<float>(r, 1) = lm[r][1];
            }
            src = tmp;
        }

        // Apply additional user-requested scaling if provided
        if (template_scale.has_value()){
            cv::Point center(image_size / 2, image_size / 2);
            src = scaleArcfaceTemplate(src, *template_scale, {center.x, center.y});
        }

        templates.push_back(src);
    }
    else {  // default mode
        // For default mode, check if we have a template for this image size
        vector<cv::Mat> src;
        auto it = _SRC_MAP.find(image_size);

        if (it == _SRC_MAP.end()){
            // If no template exists for this size, scale from 112
            const auto& base_array = _SRC_MAP.at(112); // array<Landmark5, 5>
            double scale_factor = image_size / 112.0;

            // Scale each template in the set
            for (const Landmark5& lm : base_array){
                cv::Mat templ(5, 2, CV_32F);
                for (int r = 0; r < 5; ++r){
                    templ.at<float>(r, 0) = lm[r][0];
                    templ.at<float>(r, 1) = lm[r][1];
                }

                cv::Point center(56, 56);
                cv::Mat scaled_template = scaleArcfaceTemplate(templ, scale_factor, {center.x, center.y});

                // Adjust center point for the new image size
                cv::Point new_center(image_size / 2, image_size / 2);
                cv::Mat center_old = (cv::Mat_<float>(1,2) << 56.f, 56.f);
                cv::Mat center_new = (cv::Mat_<float>(1,2) << image_size/2.f, image_size/2.f);
                cv::Mat center_old_repeat, center_new_repeat;
                cv::repeat(center_old, scaled_template.rows, 1, center_old_repeat);
                cv::repeat(center_new, scaled_template.rows, 1, center_new_repeat);
                scaled_template = scaled_template - center_old_repeat + center_new_repeat;


                src.push_back(scaled_template);
            }
        }
        else{
            auto it = _SRC_MAP.find(image_size);
            const auto& lm_array = it->second; // array<Landmark5, 5>
            for (const Landmark5& lm : lm_array){
                cv::Mat templ_mat(5, 2, CV_32F);
                for (int r = 0; r < 5; ++r){
                    templ_mat.at<float>(r, 0) = lm[r][0];
                    templ_mat.at<float>(r, 1) = lm[r][1];
                }
                src.push_back(templ_mat);
            }
        }

        // Apply additional user-requested scaling if provided
        if (template_scale.has_value()){
            vector<cv::Mat> scaled_src;
            cv::Point center(image_size / 2, image_size / 2);
            for (const cv::Mat& templ : src){
                cv::Mat scaled_template = scaleArcfaceTemplate(templ, *template_scale, {center.x, center.y});
                scaled_src.push_back(scaled_template);
            }
            src = scaled_src;
        }

        templates = src;
    }

    // Find the best matching template
    for (size_t i = 0; i < templates.size(); ++i){
        const cv::Mat& templ = templates[i];

        // Estimate similarity transform
        tform = cv::estimateAffinePartial2D(landmark, templ);
        if (tform.empty()) continue;

        cv::Mat M = tform.clone();  // 2x3

        cv::Mat results = (M * lmk_tran.t()).t();  // 5x2

        cv::Mat diff = results - templ;
        cv::Mat sq;
        cv::pow(diff, 2, sq);

        cv::Mat sum_row;
        cv::reduce(sq, sum_row, 1, cv::REDUCE_SUM);

        cv::Mat dist;
        cv::sqrt(sum_row, dist);

        double error = cv::sum(dist)[0];

        if (error < min_error){
            min_error = error;
            min_M = M.clone();
            min_index = static_cast<int>(i);
        }
    }

    return {min_M, min_index};
}


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
    cv::Mat homogeneous_landmarks;
    homogeneous_landmarks.convertTo(homogeneous_landmarks, CV_32F);
    transformed = homogeneous_landmarks * transformation_matrix.t();
    return transformed; 
}


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
    validateInputs(image, det, landmarks, template_scale);

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
