#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include "../tensor.h"
// #include "../utils.cuh"

// const int kCudaThreadsNum = 512;
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void relu_forward_gpu(float* input, float* output, int size) {
    CUDA_KERNEL_LOOP(i, size)
        output[i] = input[i] > 0 ? input[i] : 0;
}

__global__ void relu_backward_gpu(float* input, float* output, float* grad_output, float* grad_input, int size) {
    CUDA_KERNEL_LOOP(i, size) {
       grad_input[i] = input[i] > 0 ? grad_output[i] : 0;
    }
}

void relu_forward_wrapper(Tensor &input, Tensor &output) {
    assert(input.size() == output.size());
    relu_forward_gpu<<<CudaGetBlocks(input.size()), kCudaThreadsNum>>>(input.data, output.data, input.size());
}

void relu_backward_wrapper(Tensor &input, Tensor &output, Tensor &grad_output, Tensor &grad_input) {
    assert(input.size() == output.size() && input.size() == grad_output.size() && input.size() == grad_input.size());
    relu_backward_gpu<<<CudaGetBlocks(input.size()), kCudaThreadsNum>>>(input.data, output.data, grad_output.data, grad_input.data, input.size());
}