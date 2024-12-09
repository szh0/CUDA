#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include "tensor.h"

// const int kCudaThreadsNum = 512;
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)


__global__ void zero_init_gpu(float* input, int size) {
    CUDA_KERNEL_LOOP(i, size)
        input[i] = 0.0f;
}

void zero_init(Tensor &input) {
    if(input.device == "cpu") {
        for(int i = 0 ; i < input.size() ; i++)
            input.data[i] = 0.0f;
    }
    if(input.device == "gpu") {
        zero_init_gpu<<<CudaGetBlocks(input.size()), kCudaThreadsNum>>>(input.data, input.size());
        cudaDeviceSynchronize();
    }
}