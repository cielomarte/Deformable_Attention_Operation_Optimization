#include <opencv2/opencv.hpp>

void convolution3D(const std::vector<cv::Mat>& input, const cv::Mat& kernel, std::vector<cv::Mat>& output) {
    int numSlices = input.size();
    output.resize(numSlices);

    for (int i = 0; i < numSlices; ++i) {
        cv::filter2D(input[i], output[i], -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    }
}
