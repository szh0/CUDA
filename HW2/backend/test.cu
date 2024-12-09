#include "cuda_runtime.h"
#include "gtest/gtest.h"

// Unit Test for Fully Connected Layer (fc) and its Backward (fc_backward)
TEST(FullyConnectedLayerTest, ForwardBackwardTest) {
    const int N = 2; // Batch size
    const int c_in = 3; // Input features
    const int c_out = 2; // Output features

    float h_input[N * c_in] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float h_weight[c_in * c_out] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    float h_bias[c_out] = {0.1f, 0.2f};
    float h_output[N * c_out] = {0.0f};

    float *d_input, *d_output, *d_weight, *d_bias;
    cudaMalloc(&d_input, N * c_in * sizeof(float));
    cudaMalloc(&d_output, N * c_out * sizeof(float));
    cudaMalloc(&d_weight, c_in * c_out * sizeof(float));
    cudaMalloc(&d_bias, c_out * sizeof(float));

    cudaMemcpy(d_input, h_input, N * c_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, c_in * c_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, c_out * sizeof(float), cudaMemcpyHostToDevice);

    // Forward pass
    fc(d_input, d_output, d_weight, d_bias, N, c_in, c_out);
    
    float h_output_result[N * c_out];
    cudaMemcpy(h_output_result, d_output, N * c_out * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check the output
    EXPECT_NEAR(h_output_result[0], 1.4f, 1e-5);
    EXPECT_NEAR(h_output_result[1], 3.2f, 1e-5);

    // Backward pass
    float h_grad_output[N * c_out] = {0.1f, 0.2f, 0.3f, 0.4f}; // Arbitrary gradient
    float h_grad_input[N * c_in] = {0.0f};
    float h_grad_weight[c_in * c_out] = {0.0f};
    float h_grad_bias[c_out] = {0.0f};

    float *d_grad_output, *d_grad_input, *d_grad_weight, *d_grad_bias;
    cudaMalloc(&d_grad_output, N * c_out * sizeof(float));
    cudaMalloc(&d_grad_input, N * c_in * sizeof(float));
    cudaMalloc(&d_grad_weight, c_in * c_out * sizeof(float));
    cudaMalloc(&d_grad_bias, c_out * sizeof(float));

    cudaMemcpy(d_grad_output, h_grad_output, N * c_out * sizeof(float), cudaMemcpyHostToDevice);

    // Backward function
    fc_backward(d_input, d_output, d_weight, d_bias, d_grad_input, d_grad_output, d_grad_weight, d_grad_bias, N, c_in, c_out);

    float h_grad_input_result[N * c_in];
    cudaMemcpy(h_grad_input_result, d_grad_input, N * c_in * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check the gradients
    EXPECT_NEAR(h_grad_input_result[0], 0.14f, 1e-5);
    EXPECT_NEAR(h_grad_input_result[1], 0.34f, 1e-5);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_grad_weight);
    cudaFree(d_grad_bias);
}

// Unit Test for Convolution Layer (conv)
TEST(ConvolutionLayerTest, ForwardBackwardTest) {
    const int N = 1; // Batch size
    const int c_in = 1; // Number of input channels
    const int c_out = 1; // Number of output channels
    const int h_in = 5; // Input height
    const int w_in = 5; // Input width
    const int kernel_size = 3; // Kernel size
    const int stride = 1; // Stride
    const int padding = 0; // Padding

    float h_input[N * c_in * h_in * w_in] = {
        1, 2, 3, 0, 0,
        4, 5, 6, 0, 0,
        7, 8, 9, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };

    float h_kernel[c_out * c_in * kernel_size * kernel_size] = {
        1, 0, 1,
        0, 1, 0,
        1, 0, 1
    };

    float h_bias[c_out] = {0.0f};
    float h_output[N * c_out * ((h_in - kernel_size) / stride + 1) * ((w_in - kernel_size) / stride + 1)] = {0.0f};

    float *d_input, *d_output, *d_kernel, *d_bias;
    cudaMalloc(&d_input, N * c_in * h_in * w_in * sizeof(float));
    cudaMalloc(&d_output, N * c_out * ((h_in - kernel_size) / stride + 1) * ((w_in - kernel_size) / stride + 1) * sizeof(float));
    cudaMalloc(&d_kernel, c_out * c_in * kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_bias, c_out * sizeof(float));

    cudaMemcpy(d_input, h_input, N * c_in * h_in * w_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, c_out * c_in * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, c_out * sizeof(float), cudaMemcpyHostToDevice);

    // Forward pass
    conv(d_input, d_output, d_kernel, d_bias, N, c_in, c_out, h_in, w_in, kernel_size, stride, padding);
    
    float h_output_result[N * c_out * ((h_in - kernel_size) / stride + 1) * ((w_in - kernel_size) / stride + 1)];
    cudaMemcpy(h_output_result, d_output, N * c_out * ((h_in - kernel_size) / stride + 1) * ((w_in - kernel_size) / stride + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check the output
    EXPECT_NEAR(h_output_result[0], 45.0f, 1e-5);

    // Backward pass (similarly as done in the forward pass)
    // Add similar checks for gradients if implemented

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_bias);
}