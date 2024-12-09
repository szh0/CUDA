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

__global__ void maxpool_forward_gpu(const int total_size,const float *input, float* output, float *mask, 
                            const int c_in,const int h_in, const int w_in, const int h_out, const int w_out,
                            const int kernel_size, const int stride, const int padding){
    // output: (N, c_in, h_out, w_out)
    // mask: (N, c_in, h_in, w_in)
    // input: (N, c_in, h_in, w_in)
    CUDA_KERNEL_LOOP(i, total_size) {
        // calculate output index
        int n = i / (c_in * h_out * w_out);
        int c = (i / (h_out * w_out)) % c_in;
        int h = (i / w_out) % h_out;
        int w = i % w_out;
        int h_start = max(0, h * stride - padding);
        int w_start = max(0, w * stride - padding);
        int h_end = min(h * stride - padding + kernel_size, h_in);
        int w_end = min(w * stride - padding + kernel_size, w_in);
        float maxval = - __FLT_MAX__;
        int maxidx = -1;
        // 遍历卷积核
        for (int kh = h_start; kh < h_end; ++kh) {
            for (int kw = w_start; kw < w_end; ++kw) {
                int offset = n * c_in * h_in * w_in + c * h_in * w_in + kh * w_in + kw;
                if (input[offset] > maxval) {
                    maxval = input[offset];
                    maxidx = offset;
                }
            }
        }
        // 计算输出值和掩码
        output[i] = maxval;
        mask[maxidx] = 1.0f;
    }
}
__global__ void maxpool_backward_gpu(const int total_size, const float *input, const float* output,
                            const float *mask, const float *grad_output, float* grad_input, 
                            const int c_in, const int h_in, const int w_in, const int h_out, const int w_out,
                            const int kernel_size, const int stride, const int padding){
    // output: (N, c_in, h_out, w_out)
    // mask: (N, c_in, h_in, w_in)
    // input: (N, c_in, h_in, w_in)
    CUDA_KERNEL_LOOP(i, total_size) {
        // calculate output index
        int n = i / (c_in * h_out * w_out);
        int c = (i / (h_out * w_out)) % c_in;
        int h = (i / w_out) % h_out;
        int w = i % w_out;
        int h_start = max(0, h * stride - padding);
        int w_start = max(0, w * stride - padding);
        int h_end = min(h * stride - padding + kernel_size, h_in);
        int w_end = min(w * stride - padding + kernel_size, w_in);
        // 遍历卷积核
        for (int kh = h_start; kh < h_end; ++kh) {
            for (int kw = w_start; kw < w_end; ++kw) {
                int offset = n * c_in * h_in * w_in + c * h_in * w_in + kh * w_in + kw;
                if (mask[offset] == 1.0f) {
                    grad_input[offset] = grad_output[i];
                    break;
                }
            }
        }
    }
}

void maxpool(float *input, float *output, float *mask, 
            const int N, const int c_in, const int h_in, const int w_in, 
            const int kernel_size, const int stride, const int padding){
    // 计算输出的长宽
    int h_out_ = (h_in + 2 * padding - kernel_size) / stride + 1;
    int w_out_ = (w_in + 2 * padding - kernel_size) / stride + 1;
    int total_size = N * c_in * h_out_ * w_out_;

    maxpool_forward_gpu<<<CudaGetBlocks(total_size), kCudaThreadsNum>>>(
        total_size, input, output, mask, 
        c_in, h_in, w_in, h_out_, w_out_, 
        kernel_size, stride, padding
    );
}

void maxpool_backward(float *input, float *output, float *mask, float* grad_input, const float *grad_output, 
                    const int N, const int c_in, const int h_in, const int w_in, 
                    const int kernel_size, const int stride, const int padding){
    // 计算输出的长宽   
    int h_out_ = (h_in + 2 * padding - kernel_size) / stride + 1;
    int w_out_ = (w_in + 2 * padding - kernel_size) / stride + 1;
    int total_size = N * c_in * h_out_ * w_out_;

    maxpool_backward_gpu<<<CudaGetBlocks(total_size), kCudaThreadsNum>>>(
        total_size, input, output, mask, grad_output, grad_input, 
        c_in, h_in, w_in, h_out_, w_out_, 
        kernel_size, stride, padding
    );

}

void maxpool_forward_wrapper(Tensor &input, Tensor &output, Tensor &mask, 
                    const int kernel_size, const int stride, const int padding){
    const int N = input.shape[0];
    const int c_in = input.shape[1];
    const int h_in = input.shape[2];
    const int w_in = input.shape[3];
    const int h_out_ = (h_in + 2 * padding - kernel_size) / stride + 1;
    const int w_out_ = (w_in + 2 * padding - kernel_size) / stride + 1;
    assert(output.shape[0] == N && output.shape[1] == c_in && output.shape[2] == h_out_ && output.shape[3] == w_out_);
    // assert(mask.shape[0] == N && mask.shape[1] == c_in && mask.shape[2] == h_in && mask.shape[3] == w_in);
    maxpool(input.data, output.data, mask.data,
            N, c_in, h_in, w_in,
            kernel_size, stride, padding);
}

void maxpool_backward_wrapper(Tensor &input, Tensor &output, Tensor &mask, Tensor &grad_input, const Tensor &grad_output, 
                    const int kernel_size, const int stride, const int padding){
    const int N = input.shape[0];
    const int c_in = input.shape[1];
    const int h_in = input.shape[2];
    const int w_in = input.shape[3];
    const int h_out_ = (h_in + 2 * padding - kernel_size) / stride + 1;
    const int w_out_ = (w_in + 2 * padding - kernel_size) / stride + 1;
    assert(output.shape[0] == N && output.shape[1] == c_in && output.shape[2] == h_out_ && output.shape[3] == w_out_);
    // assert(mask.shape[0] == N && mask.shape[1] == c_in && mask.shape[2] == h_in && mask.shape[3] == w_in);
    assert(grad_input.shape[0] == N && grad_input.shape[1] == c_in && grad_input.shape[2] == h_in && grad_input.shape[3] == w_in);
    maxpool_backward(input.data, output.data, mask.data, grad_input.data, grad_output.data,
            N, c_in, h_in, w_in,
            kernel_size, stride, padding);
}