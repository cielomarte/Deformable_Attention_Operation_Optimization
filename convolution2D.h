#ifndef CONVOLUTION2D_H
#define CONVOLUTION2D_H

#include <opencv2/opencv.hpp>

void convolution2D(const cv::Mat& input, const cv::Mat& kernel, cv::Mat& output);

#endif // CONVOLUTION2D_H
