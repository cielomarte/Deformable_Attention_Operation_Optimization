#ifndef CONVOLUTION3D_H
#define CONVOLUTION3D_H

#include <opencv2/opencv.hpp>
#include <vector>

void convolution3D(const std::vector<cv::Mat>& input, const cv::Mat& kernel, std::vector<cv::Mat>& output);

#endif // CONVOLUTION3D_H
