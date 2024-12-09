#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <cstring>


#pragma once
// const int kCudaThreadsNum = 512;
// inline int CudaGetBlocks(int N) { 
//     return (N + kCudaThreadsNum - 1) / kCudaThreadsNum; 
// }

class Tensor {
//private:
public:
    std::vector<int> shape;
    int total_size;
    float* data;
    std::string device;
    // bool is_on_gpu;


    Tensor(const std::vector<int>& shape, const std::string device) : shape(shape), device(device) {
        total_size = 1;
        data = nullptr;
        for (auto dim : shape) {
            total_size *= dim;
        }
        if (device == "gpu") {
            cudaMalloc(&data, total_size * sizeof(float));
        } else {
            data = new float[total_size];
        }
    }

    Tensor(const std::vector<int>& shape, const std::string device, float* data) : shape(shape), device(device), data(data) {
        total_size = 1;
        for (auto dim : shape) {
            total_size *= dim;
        }

    }
    
    ~Tensor() {
        if(data) {
            if (device == "gpu") {
                cudaFree(data);
            } else {
                delete[] data;
            }
        }
    }


    float* get_data() {
        return data;
    }
    
    void set_data(float* data){
        this->data = data;
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
        Tensor tensor(this->shape, "cpu");
        if(this->device == "gpu"){
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
        Tensor tensor(this->shape, "gpu");
        if(this->device == "gpu"){
            cudaMemcpy(tensor.data, this->data, this->size() * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        else {
            cudaMemcpy(tensor.data, this->data, this->size() * sizeof(float), cudaMemcpyHostToDevice);
        }
        return tensor;
    }
    void to_cpu() {
        if(device == "gpu") {
            float* data_ = new float[size()];
            cudaMemcpy(data_, data, size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(data);
            cudaDeviceSynchronize();
            data = data_;
            device = "cpu";
        }
    }
    void to_gpu() {
        if(device == "cpu") {
            float* data_ = nullptr;
            cudaMalloc(&data_, size() * sizeof(float));
            cudaMemcpy(data_, data, size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            delete[] data;
            data = data_;
            device = "gpu";
        }
    }

    void print_data() {
        if(device == "gpu"){
            to_cpu(); // Convert to CPU to print data
            for (int i = 0; i < total_size; i++) {
                printf("%lf ", data[i]);
            }
            printf("\n");
            to_gpu();
        }
        else {
            for (int i = 0; i < total_size; i++) {
                printf("%lf ", data[i]);
            }
            printf("\n");
        }
    }

    bool operator==(Tensor& other) const {
        if (shape != other.shape) {
            for(auto i : shape) {
                printf("%d ", i);
            }
            printf("\n");
            for(auto i : other.shape) {
                printf("%d ", i);
            }
            printf("\n");

            printf("Shape not equal\n");
            return false; // 比较形状
        }
        if (total_size != other.total_size) {
            printf("Total size not equal\n");
            return false; // 比较总大小
        }
        if (device != other.device) {
            printf("Device not equal\n");
            return false; // 比较设备
        }
        
        //must be on the same device
        for(int i = 0; i < total_size; i++) {
            if(fabs(data[i] - other.data[i]) > 1e-6 ) {
                printf("Data not equal\n");
                return false;
            }
        }

        return true;
    }
};


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

Tensor* numpy_to_tensor(py::array_t<float> array){
    if (!(array.flags() & py::array::c_style)) {
        throw std::runtime_error("Numpy array must be C-contiguous");
    }
    
    py::buffer_info buf = array.request();
    std::vector<int> shape;
    for (auto len : buf.shape) {
        shape.push_back(static_cast<int>(len));
    }
    Tensor* tensor = new Tensor(shape, "gpu");
    cudaMemcpy(tensor->data, buf.ptr, sizeof(float) * tensor->size(), cudaMemcpyHostToDevice);
    return tensor; 
}

py::array_t<float> tensor_to_numpy(Tensor* tensor){
    py::array_t<float> array{tensor->shape};
    py::buffer_info buf = array.request();
    cudaMemcpy(buf.ptr, tensor->data, sizeof(float) * tensor->size(), cudaMemcpyDeviceToHost);
    return array;
}