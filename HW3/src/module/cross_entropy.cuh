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

__global__ void inter_rep_gpu(const float* input, const float *index, float* output, const int n, const int m){
    CUDA_KERNEL_LOOP(i, n) {
        output[i] = input[i * m + (int)(index[i])];
    }
}

__global__ void inter_sub_gpu(float* input, const float *index, const int n, const int m){
    CUDA_KERNEL_LOOP(i, n) {
        input[i * m + (int)index[i]] -= 1;
    }
}

void inter_rep(const float* input, const float *index, float* output, const int n, const int m){
    inter_rep_gpu<<<CudaGetBlocks(n), kCudaThreadsNum>>>(input, index, output, n, m);
}

void inter_sub(float* input, const float *index, const int n, const int m){
    inter_sub_gpu<<<CudaGetBlocks(n), kCudaThreadsNum>>>(input, index, n, m);
}

//output = -sum(target * log(input), 1)
void cross_entropy(float* input, float* target, float* input_loss, const int N, int label_num){
    // input: (N, label_num)
    // target: (N,)
    // output: loss
    thrust::device_vector<float> input_vec(N);
    thrust::device_ptr<float> loss(input_loss);

    inter_rep(input, target, thrust::raw_pointer_cast(input_vec.data()), N, label_num);
    thrust::transform(input_vec.begin(), input_vec.end(), input_vec.begin(), Sneglog());

    *loss = thrust::reduce(input_vec.begin(), input_vec.end(), 0.0f, thrust::plus<float>()) / N;
}

// with softmax
void cross_entropy_backward(float* input, float* grad_prob, float* target, float* loss, float* grad_input, 
                            const int N, int label_num){
    // input: (N, label_num)
    // target: (N,)
    thrust::device_ptr<float> prob(grad_prob);
    thrust::device_ptr<float> input_ptr(grad_input);
    thrust::copy(prob, prob + N*label_num, input_ptr);
    inter_sub(grad_input, target, N, label_num);
    thrust::transform(input_ptr, input_ptr + N*label_num, input_ptr, Sdiv(N));
}

void cross_entropy_forward_wrapper(Tensor &input, Tensor &target, Tensor &output){
    const int N = input.shape[0];
    int label_num = input.shape[1];
    cross_entropy(input.data, target.data, output.data, input.shape[0], label_num);
}

//with softmax
void cross_entropy_backward_wrapper(Tensor &input, Tensor &grad_prob, Tensor &target, Tensor &loss, Tensor &grad_input){
    const int N = input.shape[0];
    int label_num = input.shape[1];
    cross_entropy_backward(input.data, grad_prob.data, target.data, loss.data, grad_input.data, N, label_num);
}