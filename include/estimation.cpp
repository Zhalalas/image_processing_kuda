#include "estimation.h"
#include "resizer.h"        // for scaleArcfaceTemplate
#include "const.h"  
#include "scale.h"        // for ARC_FACE_TEMPLATE_MAP, _SRC_MAP
#include <opencv2/imgproc.hpp>   // for cv::estimateAffinePartial2D, cv::pow, etc.
#include <opencv2/core.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <limits>

using namespace std;

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
