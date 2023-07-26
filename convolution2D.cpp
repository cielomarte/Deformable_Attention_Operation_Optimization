#include <opencv2/opencv.hpp>


void convolution2D(const cv::Mat& input, const cv::Mat& kernel, cv::Mat& output) {
    // Perform convolution using OpenCV's filter2D function
    cv::filter2D(input, output, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
}
