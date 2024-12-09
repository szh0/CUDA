#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
// #include <device_launch_parameters.h>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include "../tensor.h"
// #include "../utils.cuh"

// const int kCudaThreadsNum = 512;
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

void softmax(float* input, float* output, const int N, int label_num) {
    // softmax
    // input: (N, label_num)
    // output: (N, label_num)    
    for (int i = 0; i < N; i++) {
        thrust::device_ptr<float> input_ptr(input + i * label_num);
        thrust::device_ptr<float> output_ptr(output + i * label_num);
        
        // 使用 thrust::reduce 计算最大值并存储到 max   
        float max = thrust::reduce(input_ptr, input_ptr + label_num, -__FLT_MAX__, thrust::maximum<float>());
        thrust::transform(input_ptr, input_ptr + label_num, output_ptr, Ssub(max));
        // 使用 thrust::transform 计算指数并存储到 output
        thrust::transform(input_ptr, input_ptr + label_num, output_ptr, Sexp());
        
        // 使用 thrust::reduce 计算和并存储结果到 sum
        float sum = thrust::reduce(output_ptr, output_ptr + label_num, 0.0f, thrust::plus<float>());
        
        if (sum > 0.0f) {  // 确保 sum 大于零
            thrust::transform(output_ptr, output_ptr + label_num, output_ptr, Sdiv(sum));
        } else if (sum < 0.0f) {
            // std::cerr << "Error: Sum is negative, cannot proceed with softmax." << std::endl;
            thrust::fill(output_ptr, output_ptr + label_num, 0.0f);  // 将输出设置为0
        } else {
            thrust::fill(output_ptr, output_ptr + label_num, 1.0f / label_num);  // 如果 sum 为零，均匀分配
        }
    }
}

void softmax_forward_wrapper(Tensor& input, Tensor& output) {
    const int N = input.shape[0];
    const int label_num = input.shape[1];
    softmax(input.data, output.data, N, label_num);
}