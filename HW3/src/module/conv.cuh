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

__global__ void im2col_gpu(const int total_size, const float* im, float *col,
                            const int c_in, const int h_in, const int w_in, 
                            const int h_out, const int w_out, 
                            const int kernel_size, const int stride, const int padding){
    // im: (c_in, h_in, w_in)
    // col: (c_in * kernel_size * kernel_size, h_out * w_out)
    CUDA_KERNEL_LOOP(i, total_size) {
        // calculate output index
        int c = i / (h_out * w_out);
        int h = (i / w_out) % h_out;
        int w = i % w_out;
        int h_start = h * stride - padding;
        int w_start = w * stride - padding;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;
        //set offset to the start of the corresponding column
        float* data_col_ptr = col;
        data_col_ptr += c * kernel_size * kernel_size * h_out * w_out + h * w_out + w;
        const float* data_im_ptr = im;
        data_im_ptr += c * h_in * w_in + h_start * w_in  + w_start;
        for (int kh = h_start; kh < h_end; ++kh) {
            for (int kw = w_start; kw < w_end; ++kw) {
                if(kh >=0 && kh < h_in && kw >= 0 && kw < w_in){
                    *data_col_ptr = data_im_ptr[(kh - h_start) * w_in + (kw - w_start)];
                }
                else {
                    *data_col_ptr = 0.0f;
                }
                // offset to the next element in the column
                data_col_ptr += h_out * w_out;
            }
        }
    }
}

__global__ void col2im_gpu(const int total_size, float* im, const float *col, 
                            const int c_in, const int h_in, const int w_in, 
                            const int h_out, const int w_out, 
                            const int kernel_size, const int stride, const int padding){
    // h_out is height of col
    CUDA_KERNEL_LOOP(i, total_size) {
        // 计算输出的索引
        int c = i / (h_in * w_in);
        int h = (i / w_in) % h_in;
        int w = i % w_in;
        //set offset to the start of the corresponding column
        const float *data_col_ptr = col;
        data_col_ptr += c * kernel_size * kernel_size * h_out * w_out;
        float *data_im_ptr = im;
        data_im_ptr += c * w_in * h_in + h * w_in + w;
        
        for (int kh = h + padding; kh > h + padding - kernel_size; kh--) {
            for (int kw = w + padding; kw > w + padding - kernel_size; kw--) {
                
                if(kh % stride == 0 && kh % stride == 0 
                    && kh >= 0 && kh/stride < h_in && kw >= 0 && kw/stride < w_in){

                    *data_im_ptr += data_col_ptr[kh / stride * w_out + kw / stride];
                }

                data_col_ptr += h_out * w_out;

            }
        }
    }
}

void im2col(const float* im, float* col, const int c_in, const int h_in, const int w_in, 
            const int kernel_size, const int stride, const int padding){
    // im: (c_in, h_in, w_in)
    // col: (c_in * kernel_size * kernel_size, h_out * w_out)
    const int h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
    const int w_out = (w_in + 2 * padding - kernel_size) / stride + 1;
    const int total_size = c_in * h_out * w_out;
    im2col_gpu<<<CudaGetBlocks(total_size), kCudaThreadsNum>>>(
        total_size, im, col, 
        c_in, h_in, w_in,
        h_out, w_out, 
        kernel_size, stride, padding
    );
}

void col2im(float* im, const float* col, const int c_in, const int h_in, const int w_in, 
            const int kernel_size, const int stride, const int padding){
    // col: (c_in * kernel_size * kernel_size, h_out * w_out)
    // im: (c_in, h_in, w_in)    
    const int h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
    const int w_out = (w_in + 2 * padding - kernel_size) / stride + 1;
    const int total_size = c_in * h_out * w_out;
    col2im_gpu<<<CudaGetBlocks(total_size), kCudaThreadsNum>>>(
        total_size, im, col, 
        c_in, h_in, w_in, 
        h_out, w_out, 
        kernel_size, stride, padding
    );
}

void conv(float* input, float* output, float* kernel, float* bias, 
        const int N, const int c_in, const int c_out, const int h_in, const int w_in, 
        const int kernel_size, const int stride, const int padding){

    // input: (N, c_in, h_in, w_in)
    // output: (N, c_out, h_out, w_out)
    // kernel: (c_out, c_in, kernel_size, kernel_size)
    int h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
    int w_out = (w_in + 2 * padding - kernel_size) / stride + 1;

    thrust::device_vector<float> col(c_in * kernel_size * kernel_size * h_out * w_out);
    for (int i = 0; i < N; i++) {
        im2col(
            input + i * c_in * h_in * w_in, thrust::raw_pointer_cast(col.data()), 
            c_in, h_in, w_in, 
            kernel_size, stride, padding
        );

        cudaDeviceSynchronize();

        gemm_gpu_NN(kernel, thrust::raw_pointer_cast(col.data()), output + i * c_out * h_out * w_out, 
            c_out, c_in * kernel_size * kernel_size, h_out * w_out);

        ma_row(output + i * c_out * h_out * w_out, bias, c_out, h_out * w_out);
    }

}


void conv_backward(float* input, float* output, float* kernel, float* bias, 
                    float* grad_input, float* grad_output, float* grad_kernel, float* grad_bias, 
                    const int N, const int c_in, const int c_out, const int h_in, const int w_in,   
                    const int kernel_size, const int stride, const int padding){
    // input: (N, c_in, h_in, w_in)
    // output: (N, c_out, h_out, w_out)
    // kernel: (c_out, c_in, kernel_size, kernel_size)
    // col: (c_in * kernel_size * kernel_size, h_out * w_out)
    // grad_input: (N, c_in, h_in, w_in)
    // grad_output: (N, c_out, h_out, w_out)
    // grad_kernel: (c_out, c_in, kernel_size, kernel_size)
    // grad_bias: (c_out, )

    int h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
    int w_out = (w_in + 2 * padding - kernel_size) / stride + 1;
    thrust::device_vector<float>  grad_col(c_in * kernel_size * kernel_size * h_out * w_out);
    thrust::device_vector<float> col(c_in * kernel_size * kernel_size * h_out * w_out);
    thrust::device_ptr<float> grad_output_ptr(grad_output);
    thrust::device_vector<float> grad_bias_vec(c_out);// 偏置梯度向量
    for (int i = 0; i < N; i++) {
        // 计算梯度

        // 计算卷积核梯度
        gemm_gpu_TN(kernel, grad_output + i * c_out * h_out * w_out, thrust::raw_pointer_cast(grad_col.data()), 
                    c_in * kernel_size * kernel_size, c_out, h_out * w_out);
        // 计算输入梯度
        col2im(grad_input + i * c_in * h_in * w_in, thrust::raw_pointer_cast(grad_col.data()), 
            c_in, h_in, w_in, kernel_size, stride, padding);

        cudaDeviceSynchronize();

        im2col(input + i * c_in * h_in * w_in, thrust::raw_pointer_cast(col.data()), 
            c_in, h_in, w_in, kernel_size, stride, padding);

        cudaDeviceSynchronize();

        gemm_gpu_NT(grad_output + i * c_out * h_out * w_out, thrust::raw_pointer_cast(col.data()), grad_kernel, 
            c_out, h_out * w_out, c_in * kernel_size * kernel_size, 1.0f, 1.0f);
        
        // 计算偏置梯度
        for (int j = 0; j < c_out; j++) {
            grad_bias_vec[j] += thrust::reduce(
                grad_output_ptr + i * c_out * h_out * w_out + j * h_out * w_out, 
                grad_output_ptr + i * c_out * h_out * w_out + j * h_out * w_out + h_out * w_out, 0.0f, thrust::plus<float>());
        }
    }
    thrust::copy(grad_bias_vec.begin(), grad_bias_vec.end(), grad_bias);

}

void conv_forward_wrapper(Tensor& input, Tensor& output, Tensor& kernel, Tensor& bias, 
                          const int kernel_size, const int stride, const int padding){
    
    const int N = input.shape[0];
    const int c_in = input.shape[1];
    const int h_in = input.shape[2];
    const int w_in = input.shape[3];
    const int c_out = output.shape[1];

    conv(input.data, output.data, kernel.data, bias.data, 
        N, c_in, c_out, h_in, w_in, 
        kernel_size, stride, padding);
}

void conv_backward_wrapper(Tensor& input, Tensor& output, Tensor& kernel, Tensor& bias, 
                           Tensor& grad_input, Tensor& grad_output, Tensor& grad_kernel, Tensor& grad_bias, 
                           const int kernel_size, const int stride, const int padding){
    
    const int N = input.shape[0];
    const int c_in = input.shape[1];
    const int h_in = input.shape[2];
    const int w_in = input.shape[3];
    const int c_out = output.shape[1];

    conv_backward(input.data, output.data, kernel.data, bias.data, 
        grad_input.data, grad_output.data, grad_kernel.data, grad_bias.data, 
        N, c_in, c_out, h_in, w_in, 
        kernel_size, stride, padding);

}
