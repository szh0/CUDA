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

void fc(float* input, float *output, float* weight, float* bias, 
        const int N, const int c_in, const int c_out){

    //X * Weights
    gemm_gpu_NN(input, weight, output, N, c_in, c_out);
    ma_col(output, bias, N, c_out);
}

void fc_backward(float* input, float *output, float* weight, float* bias, 
                float* grad_input, float* grad_output, float* grad_weight, float* grad_bias, 
                const int N, const int c_in, const int c_out){
    
    //grad_input = grad_output * W^T
    //grad_output: (N, c_out)
    //grad_input: (N, c_in)
    gemm_gpu_NT(grad_output, weight, grad_input, N, c_out, c_in);
    
    //grad_weight = X^T * grad_output
    //grad_weight: (c_in, c_out)
    //grad_output: (N, c_out)
    gemm_gpu_TN(input, grad_output, grad_weight, c_in, N, c_out);
    // sumcol_gpu(grad_output, grad_bias, N, out_features);    

    //grad_bias = sum(grad_output, 0)
    //grad_bias: (c_out, )
    //grad_output: (N, c_out)
    ms_col(grad_output, grad_bias, N, c_out);
}

void fc_forward_wrapper(Tensor &input, Tensor &output, Tensor &weight, Tensor &bias){
    const int N = input.shape[0];
    const int c_in = input.shape[1];
    const int c_out = output.shape[1];
    assert(input.shape[1] == weight.shape[0]);
    assert(weight.shape[1] == bias.shape[0]);
    assert(bias.shape[0] == c_out);
    assert(input.shape[0] == output.shape[0]);
    fc(input.data, output.data, weight.data, bias.data, N, c_in, c_out);
}

void fc_backward_wrapper(Tensor &input, Tensor &output, Tensor &weight, Tensor &bias, 
                        Tensor &grad_input, Tensor &grad_output, Tensor &grad_weight, Tensor &grad_bias){
    const int N = input.shape[0];
    const int c_in = input.shape[1];
    const int c_out = output.shape[1];
    assert(input.shape[1] == weight.shape[0]);
    assert(weight.shape[1] == bias.shape[0]);
    assert(bias.shape[0] == c_out);
    assert(input.shape[0] == output.shape[0]);
    assert(grad_input.shape[0] == N);
    assert(grad_input.shape[1] == c_in);
    assert(grad_output.shape[0] == N);
    assert(grad_output.shape[1] == c_out);
    assert(grad_weight.shape[0] == c_in);
    assert(grad_weight.shape[1] == c_out);
    assert(grad_bias.shape[0] == c_out);
    fc_backward(input.data, output.data, weight.data, bias.data, 
                grad_input.data, grad_output.data, grad_weight.data, grad_bias.data, 
                N, c_in, c_out);
}