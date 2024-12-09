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

__global__ void sigmoid_forward_gpu(float* input, float* output, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        float exp_val = expf(-input[i]);
        output[i] = 1.0f / (1.0f + exp_val);
    }
}

__global__ void sigmoid_backward_gpu(float* output, float* grad_output, float* grad_input, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        float sigmoid = output[i];
        grad_input[i] = grad_output[i] * sigmoid * (1 - sigmoid);
    }
}

void sigmoid_forward_wrapper(Tensor &input, Tensor &output) {
    sigmoid_forward_gpu<<<CudaGetBlocks(input.size()), kCudaThreadsNum>>>(input.data, output.data, input.size());
}

void sigmoid_backward_wrapper(Tensor &output, Tensor &grad_output, Tensor &grad_input) {
    sigmoid_backward_gpu<<<CudaGetBlocks(output.size()), kCudaThreadsNum>>>(output.data, grad_output.data, grad_input.data, output.size());
}