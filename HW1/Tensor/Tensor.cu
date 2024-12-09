#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <cstring>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int kCudaThreadsNum = 512;//threads in block
inline int CudaGetBlocks(int N) { 
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum; 
}

float relu_cpu(float x) {
    return x > 0 ? x : 0;
}

float sigmoid_cpu(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ 
void relu_forward_gpu(float* input, float* output, int size) {
    CUDA_KERNEL_LOOP(i, size)
        output[i] = input[i] > 0 ? input[i] : 0;
}

__global__ 
void relu_backward_gpu(float* input, float* output, float* grad_output, float* grad_input, int size) {
    CUDA_KERNEL_LOOP(i, size) {
       grad_input[i] = input[i] > 0 ? grad_output[i] : 0;
    }
}

__global__ 
void sigmoid_forward_gpu(float* input, float* output, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        float exp_val = expf(-input[i]);
        output[i] = 1.0f / (1.0f + exp_val);
    }
}

__global__ 
void sigmoid_backward_gpu(float* input, float* output, float* grad_output, float* grad_input, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        float sigmoid = output[i];
        grad_input[i] = grad_output[i] * sigmoid * (1 - sigmoid);
    }
}

class Tensor {
//private:
public:
    std::vector<int> shape;
    int total_size;
    float* data;
    bool is_on_gpu;


    Tensor(const std::vector<int>& shape, bool use_gpu) : shape(shape), is_on_gpu(use_gpu) {
        total_size = 1;
        data = nullptr;
        for (auto dim : shape) {
            total_size *= dim;
        }
        if (is_on_gpu) {
            cudaMalloc(&data, total_size * sizeof(float));
        } else {
            data = new float[total_size];
        }
    }

    Tensor(const std::vector<int>& shape, bool use_gpu, float* data) : shape(shape), is_on_gpu(use_gpu), data(data) {
        total_size = 1;
        for (auto dim : shape) {
            total_size *= dim;
        }
    }
    
    ~Tensor() {
        if(data) {
            if (is_on_gpu) {
                cudaFree(data);
            } else {
                delete[] data;
            }
        }
    }
    // size
    int size() {
        return total_size;
    }

    // index
    float& operator[](int index) {
        return data[index];
    }

    // GPU to CPU
    Tensor cpu() {
        // float* cpu_data = new float[size()];
        // if (is_on_gpu) {
        //     cudaMemcpy(cpu_data, data, sizeof(float) * size(), cudaMemcpyDeviceToHost);
        // }
        // return Tensor(shape, false, cpu_data);        
        Tensor tensor(this->shape, false);
        if(this->is_on_gpu){
            cudaMemcpy(tensor.data, this->data, this->size() * sizeof(float), cudaMemcpyDeviceToHost);
        }
        else {
            cudaMemcpy(tensor.data, this->data, this->size() * sizeof(float), cudaMemcpyHostToHost);
        }
        return tensor;
    }

    // CPU to GPU
    Tensor gpu() {
        // float* gpu_data = nullptr;
        // cudaMalloc(&gpu_data, sizeof(float) * size());
        // if (!is_on_gpu) {
        //     cudaMemcpy(gpu_data, data, sizeof(float) * size(), cudaMemcpyHostToDevice);
        // }
        // return Tensor(shape, true, gpu_data);
        Tensor tensor(this->shape, true);
        if(this->is_on_gpu){
            cudaMemcpy(tensor.data, this->data, this->size() * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        else {
            cudaMemcpy(tensor.data, this->data, this->size() * sizeof(float), cudaMemcpyHostToDevice);
        }
        return tensor;
    }
    void to_cpu() {
        if(is_on_gpu) {
            float* data_ = new float[size()];
            cudaMemcpy(data_, data, size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(data);
            data = data_;
            is_on_gpu = false;
        }
    }
    void to_gpu() {
        if(!is_on_gpu) {
            float* data_ = nullptr;
            cudaMalloc(&data_, size() * sizeof(float));
            cudaMemcpy(data_, data, size() * sizeof(float), cudaMemcpyHostToDevice);
            delete[] data;
            data = data_;
            is_on_gpu = true;
        }
    }
};

void ReLUForwardCPU(Tensor* input, Tensor* output) {
    for(int i = 0; i < input->size(); i++) {
        output->data[i] = relu_cpu(input->data[i]);
    }
}

void ReLUBackwardCPU(Tensor* input, Tensor* output,Tensor* grad_output, Tensor* grad_input) {
    for(int i = 0; i < input->size(); i++) {
        grad_input->data[i] = input->data[i] > 0 ? grad_output->data[i] : 0;
    }
}

void SigmoidForwardCPU(Tensor* input, Tensor* output) {
    for(int i = 0; i < input->size(); i++) {
        output->data[i] = sigmoid_cpu(input->data[i]);
    }
}

void SigmoidBackwardCPU(Tensor* input, Tensor* output,Tensor* grad_output, Tensor* grad_input) {
    float sigmoidval;
    for(int i = 0; i < input->size(); i++) {
        sigmoidval = output->data[i];
        grad_input->data[i] = grad_output->data[i] * sigmoidval * (1 - sigmoidval);
    }
}

void ReLUForwardGPU(Tensor* input, Tensor* output) {
    relu_forward_gpu<<<CudaGetBlocks(input->size()), kCudaThreadsNum>>>(
        input->data, 
        output->data, 
        input->size()
    );
}

void ReLUBackwardGPU(Tensor* input, Tensor* output,Tensor* grad_output, Tensor* grad_input) {
    relu_backward_gpu<<<CudaGetBlocks(input->size()), kCudaThreadsNum>>>(
        input->data, 
        output->data, 
        grad_output->data, 
        grad_input->data, 
        input->size()
    );
}

void SigmoidForwardGPU(Tensor* input, Tensor* output) {
    sigmoid_forward_gpu<<<CudaGetBlocks(input->size()), kCudaThreadsNum>>>(
        input->data, 
        output->data, 
        input->size()
    );
}

void SigmoidBackwardGPU(Tensor* input, Tensor* output,Tensor* grad_output, Tensor* grad_input) {
    sigmoid_backward_gpu<<<CudaGetBlocks(input->size()), kCudaThreadsNum>>>(
        input->data, 
        output->data, 
        grad_output->data, 
        grad_input->data, 
        input->size()
    );
}
