#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include <iostream>
#include <string>
#include <fstream>

#include "module.cuh"

float EPS = 1e-3f;

void fill(thrust::device_vector<float> &vec, std::ifstream &file) {
    int size = vec.size();
    float x = 0;
    for (int i = 0; i < size; i++) {
        file >> x;
        vec[i] = x;
    }
}

bool check(thrust::device_vector<float> &gt, thrust::device_vector<float> &res) {
    int size = gt.size();
    for (int i = 0; i < size; i++) {
        if (abs(gt[i] - res[i]) > EPS)
            return false;
    }
    return true;
}

void print(thrust::device_vector<float> &vec) {
    int size = vec.size();
    for (int i = 0; i < size; i++) {
        std::cout << vec[i] << ' ';
    }
    std::cout << '\n';
}

void test_fc() {
    std::ifstream file;
    file.open("data/fc.txt");
    int batchsize = 0, in_features = 0, out_features = 0;
    file >> batchsize >> in_features >> out_features;
    thrust::device_vector<float> input(batchsize * in_features);
    thrust::device_vector<float> output_gt(batchsize * out_features);
    thrust::device_vector<float> output_res(batchsize * out_features);
    thrust::device_vector<float> weight(out_features * in_features);
    thrust::device_vector<float> bias(out_features);
    thrust::device_vector<float> grad_output(batchsize * out_features);
    thrust::device_vector<float> grad_input_gt(batchsize * in_features);
    thrust::device_vector<float> grad_input_res(batchsize * in_features);
    thrust::device_vector<float> grad_weight_gt(out_features * in_features);
    thrust::device_vector<float> grad_weight_res(out_features * in_features);
    thrust::device_vector<float> grad_bias_gt(out_features);
    thrust::device_vector<float> grad_bias_res(out_features);
    fill(input, file);
    fill(output_gt, file);
    fill(weight, file);
    fill(bias, file);
    fill(grad_output, file);
    fill(grad_input_gt, file);
    fill(grad_weight_gt, file);
    fill(grad_bias_gt, file);
    fc(thrust::raw_pointer_cast(input.data()),
               thrust::raw_pointer_cast(output_res.data()),
               thrust::raw_pointer_cast(weight.data()),
               thrust::raw_pointer_cast(bias.data()),
               batchsize, in_features, out_features);
    fc_backward(thrust::raw_pointer_cast(input.data()),
                thrust::raw_pointer_cast(output_res.data()),
                thrust::raw_pointer_cast(weight.data()),
                thrust::raw_pointer_cast(bias.data()),
                thrust::raw_pointer_cast(grad_input_res.data()),
                thrust::raw_pointer_cast(grad_output.data()),
                thrust::raw_pointer_cast(grad_weight_res.data()),
                thrust::raw_pointer_cast(grad_bias_res.data()),
                batchsize, in_features, out_features);
    if (check(output_gt, output_res)) {
        std::cout<<"Test fc forward passed\n";
    } else {
        std::cout<<"Test fc forward failed\n";
    }
    if (check(grad_input_gt, grad_input_res) &&
        check(grad_weight_gt, grad_weight_res) &&
        check(grad_bias_gt, grad_bias_res)) {
        std::cout<<"Test fc backward passed\n";
    } else {
        std::cout<<"Test fc backward failed\n";
    }
    file.close();
}

void test_conv() {
    std::ifstream file;
    file.open("data/conv.txt");
    int batchsize = 0, channels_in = 0, channels_out = 0, height = 0, width = 0, ksize = 0, pad = 0, stride = 0;
    int height_out = 0, width_out = 0;
    file >> batchsize >> channels_in >> channels_out >> height >> width >> ksize >> pad >> stride;
    height_out = (height + 2 * pad - ksize) / stride + 1;
    width_out = (width + 2 * pad - ksize) / stride + 1;
    thrust::device_vector<float> input(batchsize * channels_in * height * width);
    thrust::device_vector<float> output_gt(batchsize * channels_out * height_out * width_out);
    thrust::device_vector<float> output_res(batchsize * channels_out * height_out * width_out);
    thrust::device_vector<float> kernel(channels_out * channels_in * ksize * ksize);
    thrust::device_vector<float> bias(channels_out);
    thrust::device_vector<float> grad_output(batchsize * channels_out * height_out * width_out);
    thrust::device_vector<float> grad_input_gt(batchsize * channels_in * height * width);
    thrust::device_vector<float> grad_input_res(batchsize * channels_in * height * width);
    thrust::device_vector<float> grad_kernel_gt(channels_out * channels_in * ksize * ksize);
    thrust::device_vector<float> grad_kernel_res(channels_out * channels_in * ksize * ksize, 0);
    thrust::device_vector<float> grad_bias_gt(channels_out);
    thrust::device_vector<float> grad_bias_res(channels_out, 0);
    fill(input, file);
    fill(output_gt, file);
    fill(kernel, file);
    fill(bias, file);
    fill(grad_output, file);
    fill(grad_input_gt, file);
    fill(grad_kernel_gt, file);
    fill(grad_bias_gt, file);
    conv(thrust::raw_pointer_cast(input.data()),
                 thrust::raw_pointer_cast(output_res.data()),
                 thrust::raw_pointer_cast(kernel.data()),
                 thrust::raw_pointer_cast(bias.data()),
                 batchsize, channels_in, channels_out, height, width, ksize, stride, pad);
    conv_backward(thrust::raw_pointer_cast(input.data()),
                  thrust::raw_pointer_cast(output_res.data()),
                  thrust::raw_pointer_cast(kernel.data()),
                  thrust::raw_pointer_cast(bias.data()),
                  thrust::raw_pointer_cast(grad_input_res.data()),
                  thrust::raw_pointer_cast(grad_output.data()),
                  thrust::raw_pointer_cast(grad_kernel_res.data()),
                  thrust::raw_pointer_cast(grad_bias_res.data()),
                  batchsize, channels_in, channels_out, height, width, ksize, stride, pad);
    if (check(output_gt, output_res)) {
        std::cout<<"Test conv forward passed\n";
    } else {
        std::cout<<"Test conv forward failed\n";
    }
    if (check(grad_input_gt, grad_input_res) &&
        check(grad_kernel_gt, grad_kernel_res) &&
        check(grad_bias_gt, grad_bias_res)) {
        std::cout<<"Test conv backward passed\n";
    } else {
        std::cout<<"Test conv backward failed\n";
    }
    file.close();
}

void test_maxpool() {
    std::ifstream file;
    file.open("data/maxpool.txt");
    int batchsize = 0, channels = 0, height = 0, width = 0, ksize = 0, pad = 0, stride = 0;
    int height_out = 0, width_out = 0;
    file >> batchsize >> channels >> height >> width >> ksize >> pad >> stride;
    height_out = (height + 2 * pad - ksize) / stride + 1;
    width_out = (width + 2 * pad - ksize) / stride + 1;
    thrust::device_vector<float> input(batchsize * channels * height * width);
    thrust::device_vector<float> output_gt(batchsize * channels * height_out * width_out);
    thrust::device_vector<float> output_res(batchsize * channels * height_out * width_out);
    thrust::device_vector<float> mask(batchsize * channels * height * width, 0);
    thrust::device_vector<float> grad_output(batchsize * channels * height_out * width_out);
    thrust::device_vector<float> grad_input_gt(batchsize * channels * height * width);
    thrust::device_vector<float> grad_input_res(batchsize * channels * height * width);
    fill(input, file);
    fill(output_gt, file);
    fill(grad_output, file);
    fill(grad_input_gt, file);
    maxpool(thrust::raw_pointer_cast(input.data()),
                    thrust::raw_pointer_cast(output_res.data()),
                    thrust::raw_pointer_cast(mask.data()),
                    batchsize, channels, height, width, ksize, stride, pad);
    maxpool_backward(thrust::raw_pointer_cast(input.data()),
                     thrust::raw_pointer_cast(output_res.data()),
                     thrust::raw_pointer_cast(mask.data()),
                     thrust::raw_pointer_cast(grad_input_res.data()),
                     thrust::raw_pointer_cast(grad_output.data()),
                     batchsize, channels, height, width, ksize, stride, pad);
    if (check(output_gt, output_res)) {
        std::cout<<"Test maxpool forward passed\n";
    } else {
        std::cout<<"Test maxpool forward failed\n";
    }
    if (check(grad_input_gt, grad_input_res)) {
        std::cout<<"Test maxpool backward passed\n";
    } else {
        std::cout<<"Test maxpool backward failed\n";
    }
    file.close();
}

void test_celoss() {
    std::ifstream file;
    file.open("data/celoss.txt");
    int batchsize = 0, labels = 0;
    file >> batchsize >> labels;
    thrust::device_vector<float> input(batchsize * labels);
    thrust::device_vector<float> prob(batchsize * labels);
    thrust::device_vector<float> target(batchsize);
    thrust::device_vector<float> loss_gt(1);
    thrust::device_vector<float> loss_res(1);
    thrust::device_vector<float> grad_input_gt(batchsize * labels);
    thrust::device_vector<float> grad_input_res(batchsize * labels);
    fill(input, file);
    fill(target, file);
    fill(loss_gt, file);
    fill(grad_input_gt, file);
    softmax(thrust::raw_pointer_cast(input.data()),
                    thrust::raw_pointer_cast(prob.data()),
                    batchsize, labels);
    cross_entropy(thrust::raw_pointer_cast(prob.data()),
                   thrust::raw_pointer_cast(target.data()),
                   thrust::raw_pointer_cast(loss_res.data()),
                   batchsize, labels);
    cross_entropy_backward(thrust::raw_pointer_cast(input.data()),
                           thrust::raw_pointer_cast(prob.data()),
                           thrust::raw_pointer_cast(target.data()),
                           thrust::raw_pointer_cast(loss_res.data()),
                           thrust::raw_pointer_cast(grad_input_res.data()),
                           batchsize, labels);
    if (check(loss_gt, loss_res)) {
        std::cout<<"Test celoss forward passed\n";
    } else {
        std::cout<<"Test celoss forward failed\n";
    }
    if (check(grad_input_gt, grad_input_res)) {
        std::cout<<"Test celoss backward passed\n";
    } else {
        std::cout<<"Test celoss backward failed\n";
    }
    file.close();
}

int main(int argc, char **argv) {
    std::string mod = argv[1];
    if (mod == "fc") {
        test_fc();
    } else if (mod == "conv") {
        test_conv();
    } else if (mod == "maxpool") {
        test_maxpool();
    } else if (mod == "celoss") {
        test_celoss();
    }
}