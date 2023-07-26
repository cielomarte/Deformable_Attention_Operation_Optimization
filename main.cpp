#include <iostream>
#include <vector>
#include <chrono>
#include "matrix_multiplication.h"
#include "deformable_attention.h"
#include "convolution2D.h"
#include "convolution3D.h"


int main() {
    const int N = 1024; // Increase matrix size to 1024 (N x N)
    const int num_heads = 8;
    const int head_dim = N / num_heads;
    const int seq_len = N;

    // Initialize matrices and deformable attention tensors (for simplicity, fill with ones)
    std::vector<float> A(N * N, 1.0f), B(N * N, 1.0f), C_scalar(N * N, 0.0f), C_avx2(N * N, 0.0f);
    std::vector<float> Q(seq_len * num_heads * head_dim, 1.0f), K(seq_len * num_heads * head_dim, 1.0f), V(seq_len * num_heads * head_dim, 1.0f), output_deformable(seq_len * num_heads * head_dim, 0.0f);

    // Benchmark scalar matrix multiplication
    auto start_scalar = std::chrono::high_resolution_clock::now();
    scalar_matrix_multiply(A, B, C_scalar, N);
    auto end_scalar = std::chrono::high_resolution_clock::now();
    auto duration_scalar = std::chrono::duration_cast<std::chrono::microseconds>(end_scalar - start_scalar);
    std::cout << "Scalar Matrix Multiplication took: " << duration_scalar.count() << " microseconds.\n";

    // Benchmark AVX2-optimized matrix multiplication
    auto start_avx2 = std::chrono::high_resolution_clock::now();
    avx2_matrix_multiply(A, B, C_avx2, N);
    auto end_avx2 = std::chrono::high_resolution_clock::now();
    auto duration_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(end_avx2 - start_avx2);
    std::cout << "AVX2-Optimized Matrix Multiplication took: " << duration_avx2.count() << " microseconds.\n";

    // Benchmark deformable attention operation
    auto start_deformable = std::chrono::high_resolution_clock::now();
    deformable_attention(Q, K, V, output_deformable, seq_len, num_heads, head_dim);
    auto end_deformable = std::chrono::high_resolution_clock::now();
    auto duration_deformable = std::chrono::duration_cast<std::chrono::microseconds>(end_deformable - start_deformable);
    std::cout << "Deformable Attention Operation took: " << duration_deformable.count() << " microseconds.\n";

    // Verify correctness (compare C_scalar, C_avx2, and output_deformable)
    // (Add verification code if needed)

    ///////
    // Initialize your input and kernel matrices here
    cv::Mat input2D = cv::Mat::ones(1000, 1000, CV_32F); // Example: 1000x1000 matrix of ones
    cv::Mat kernel2D = cv::Mat::ones(3, 3, CV_32F); // Example: 3x3 matrix of ones
    cv::Mat output2D;

    // Perform 2D convolution and measure the time
    auto start2D = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) { // Repeat 100 times for benchmarking
        convolution2D(input2D, kernel2D, output2D);
    }
    auto end2D = std::chrono::high_resolution_clock::now();
    auto duration2D = std::chrono::duration_cast<std::chrono::microseconds>(end2D - start2D);
    std::cout << "Average time for 2D convolution: " << duration2D.count() / 100.0 << " microseconds" << std::endl;

    // Initialize your input and kernel 3D tensors here
    std::vector<cv::Mat> input3D(10, cv::Mat::ones(100, 100, CV_32F)); // Example: 10 slices of 100x100 matrices of ones
    cv::Mat kernel3D = cv::Mat::ones(3, 3, CV_32F); // Example: 3x3 matrix of ones
    std::vector<cv::Mat> output3D;

    // Perform 3D convolution and measure the time
    auto start3D = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) { // Repeat 100 times for benchmarking
        convolution3D(input3D, kernel3D, output3D);
    }
    auto end3D = std::chrono::high_resolution_clock::now();
    auto duration3D = std::chrono::duration_cast<std::chrono::microseconds>(end3D - start3D);
    std::cout << "Average time for 3D convolution: " << duration3D.count() / 100.0 << " microseconds" << std::endl;

 

    return 0;
}
