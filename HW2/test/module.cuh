#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include <iostream>
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int kCudaThreadsNum = 512;//threads in block
inline int CudaGetBlocks(int N) { 
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum; 
}


struct Sexp {
    __host__ __device__ float operator()(const float x) const {
        return std::exp(x); // 使用 std::exp 计算指数
    }
};

struct Ssub {
    const float a;
    Ssub(float a) : a(a) {}
    __host__ __device__
    constexpr float operator()(const float &x) const {
        return x - a;
    }
};

struct Sdiv {
    const float a;
    Sdiv(float a) : a(a) {}
    __host__ __device__
    constexpr float operator()(const float &x) const {
        return x / a;
    }
};

struct Sneglog{
    Sneglog() {}
    __host__ __device__ float operator()(const float x) const {
        return -logf(x); 
    }
};

// C(n,m) = A(n,k) * B(k,m)
void gemm_gpu_NN(const float *A, const float *B, float *C, const int n, const int k, const int m,
                const float alf = 1.0f, const float bet = 0.0f) { 
    int lda = k;
    int ldb = m;
    int ldc = m;
    const float *alpha = &alf; 
    const float *beta = &bet; 
    // Create a handle for CUBLAS 
    cublasHandle_t handle; 
    cublasCreate(&handle); 
    // Do the actual multiplication 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc); 
    // Destroy the handle cublasDestroy(handle); 
    cublasDestroy(handle);
}

// C(n,m) = A^T(n,k) * B(k,m)
// A(k,n)
void gemm_gpu_TN(const float *A, const float *B, float *C, const int n, const int k, const int m,
                const float alf = 1.0f, const float bet = 0.0f) { 
    int lda = n;
    int ldb = m;
    int ldc = m;
    const float *alpha = &alf; 
    const float *beta = &bet; 
    // Create a handle for CUBLAS 
    cublasHandle_t handle; 
    cublasCreate(&handle); 
    // Do the actual multiplication 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc); 
    // Destroy the handle cublasDestroy(handle); 
    cublasDestroy(handle);
}

// C(n,m) = A(n,k) * B^T(k,m)
// B(m,k)
void gemm_gpu_NT(const float *A, const float *B, float *C, const int n, const int k, const int m,
                const float alf = 1.0f, const float bet = 0.0f) { 
    int lda = k;
    int ldb = k;
    int ldc = m;
    const float *alpha = &alf; 
    const float *beta = &bet; 
    // Create a handle for CUBLAS 
    cublasHandle_t handle; 
    cublasCreate(&handle); 
    // Do the actual multiplication 
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc); 
    // Destroy the handle cublasDestroy(handle); 
    cublasDestroy(handle);
}

// C(n,m) = A^T(n,k) * B^T(k,m)
// A(k,n) B(m,k)
void gemm_gpu_TT(const float *A, const float *B, float *C, const int n, const int k, const int m,
                const float alf = 1.0f, const float bet = 0.0f) {  
    int lda = n;
    int ldb = k;
    int ldc = m;
    const float *alpha = &alf; 
    const float *beta = &bet; 
    // Create a handle for CUBLAS 
    cublasHandle_t handle; 
    cublasCreate(&handle); 
    // Do the actual multiplication  
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc); 
    // Destroy the handle cublasDestroy(handle); 
    cublasDestroy(handle);
}

//matrix addition on GPU
// by column
// A(n, m) += bias(m, )
void ma_col(float *A, const float *bias, const int n, const int m) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alp = 1.0f;
    for (int i = 0; i < n; i++) {
      cublasSaxpy(handle, m, &alp, bias, 1, A + i * m, 1);
    }
    cublasDestroy(handle);
} 

// by row
// A(n, m) += bias(n, )
void ma_row(float *A, const float *bias, const int n, const int m) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alp = 1.0f;
    for (int i = 0; i < m; i++) {
      cublasSaxpy(handle, n, &alp, bias, 1, A + i, m);
    }
    cublasDestroy(handle);
} 

// Fill the matrix with random numbers on GPU 
void matrix_init(float *A, int rows, int cols) { 
    // Create a pseudo-random number generator 
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT); 
    
    // Set the seed for the random number generator using the system clock 
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock()); 
    
    // Fill the array with random numbers on the device 
    curandGenerateUniform(prng, A, rows * cols); 
    curandDestroyGenerator(prng); 
}

//matrix sum on GPU
// by column
void ms_col(const float *A, float *sum, const int n, const int m){
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alf = 1.0f;
    for (int i = 0; i < m; i++) {
        cublasSaxpy(handle, n, &alf, A + i, m, sum + i, 0);
    }
    cublasDestroy(handle);
} 

__global__ void sumcol_gpu(const float* input, float* output, const int n, const int m){
    CUDA_KERNEL_LOOP(i, m) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += input[j * m + i];
        }
        output[i] = sum;
    }
}


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

__global__ void maxpool_forward_gpu(const int total_size,const float *input, float* output, float *mask, 
                            const int c_in,const int h_in, const int w_in, const int h_out, const int w_out,
                            const int kernel_size, const int stride, const int padding){
    // output: (N, c_in, h_out, w_out)
    // mask: (N, c_in, h_in, w_in)
    // input: (N, c_in, h_in, w_in)
    CUDA_KERNEL_LOOP(i, total_size) {
        // calculate output index
        int n = i / (c_in * h_out * w_out);
        int c = (i / (h_out * w_out)) % c_in;
        int h = (i / w_out) % h_out;
        int w = i % w_out;
        int h_start = max(0, h * stride - padding);
        int w_start = max(0, w * stride - padding);
        int h_end = min(h * stride - padding + kernel_size, h_in);
        int w_end = min(w * stride - padding + kernel_size, w_in);
        float maxval = -FLT_MAX;
        int maxidx = -1;
        // 遍历卷积核
        for (int kh = h_start; kh < h_end; ++kh) {
            for (int kw = w_start; kw < w_end; ++kw) {
                int offset = n * c_in * h_in * w_in + c * h_in * w_in + kh * w_in + kw;
                if (input[offset] > maxval) {
                    maxval = input[offset];
                    maxidx = offset;
                }
            }
        }
        // 计算输出值和掩码
        output[i] = maxval;
        mask[maxidx] = 1.0f;
    }
}
__global__ void maxpool_backward_gpu(const int total_size, const float *input, const float* output,
                            const float *mask, const float *grad_output, float* grad_input, 
                            const int c_in, const int h_in, const int w_in, const int h_out, const int w_out,
                            const int kernel_size, const int stride, const int padding){
    // output: (N, c_in, h_out, w_out)
    // mask: (N, c_in, h_in, w_in)
    // input: (N, c_in, h_in, w_in)
    CUDA_KERNEL_LOOP(i, total_size) {
        // calculate output index
        int n = i / (c_in * h_out * w_out);
        int c = (i / (h_out * w_out)) % c_in;
        int h = (i / w_out) % h_out;
        int w = i % w_out;
        int h_start = max(0, h * stride - padding);
        int w_start = max(0, w * stride - padding);
        int h_end = min(h * stride - padding + kernel_size, h_in);
        int w_end = min(w * stride - padding + kernel_size, w_in);
        // 遍历卷积核
        for (int kh = h_start; kh < h_end; ++kh) {
            for (int kw = w_start; kw < w_end; ++kw) {
                int offset = n * c_in * h_in * w_in + c * h_in * w_in + kh * w_in + kw;
                if (mask[offset] == 1.0f) {
                    grad_input[offset] = grad_output[i];
                    break;
                }
            }
        }
    }
}

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



void inter_rep(const float* input, const float *index, float* output, const int n, const int m){
    inter_rep_gpu<<<CudaGetBlocks(n), kCudaThreadsNum>>>(input, index, output, n, m);
}

void inter_sub(float* input, const float *index, const int n, const int m){
    inter_sub_gpu<<<CudaGetBlocks(n), kCudaThreadsNum>>>(input, index, n, m);
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

// Y = X * W  + b
// weight reverse order of standard 
// X: (N, c_in)
// W: (c_in, c_out)
// b: (c_out, )
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
void maxpool(float *input, float *output, float *mask, 
            const int N, const int c_in, const int h_in, const int w_in, 
            const int kernel_size, const int stride, const int padding){
    // 计算输出的长宽
    int h_out_ = (h_in + 2 * padding - kernel_size) / stride + 1;
    int w_out_ = (w_in + 2 * padding - kernel_size) / stride + 1;
    int total_size = N * c_in * h_out_ * w_out_;

    maxpool_forward_gpu<<<CudaGetBlocks(total_size), kCudaThreadsNum>>>(
        total_size, input, output, mask, 
        c_in, h_in, w_in, h_out_, w_out_, 
        kernel_size, stride, padding
    );

}

void maxpool_backward(float *input, float *output, float *mask, float* grad_input, const float *grad_output, 
                    const int N, const int c_in, const int h_in, const int w_in, 
                    const int kernel_size, const int stride, const int padding){
    // 计算输出的长宽   
    int h_out_ = (h_in + 2 * padding - kernel_size) / stride + 1;
    int w_out_ = (w_in + 2 * padding - kernel_size) / stride + 1;
    int total_size = N * c_in * h_out_ * w_out_;

    maxpool_backward_gpu<<<CudaGetBlocks(total_size), kCudaThreadsNum>>>(
        total_size, input, output, mask, grad_output, grad_input, 
        c_in, h_in, w_in, h_out_, w_out_, 
        kernel_size, stride, padding
    );
}

void softmax(float* input, float* output, const int N, int label_num) {
    // softmax
    // input: (N, label_num)
    // output: (N, label_num)    
    for (int i = 0; i < N; i++) {
        thrust::device_ptr<float> input_ptr(input + i * label_num);
        thrust::device_ptr<float> output_ptr(output + i * label_num);
        
        // 使用 thrust::reduce 计算最大值并存储到 max   
        float max = thrust::reduce(input_ptr, input_ptr + label_num, -FLT_MAX, thrust::maximum<float>());
        thrust::transform(input_ptr, input_ptr + label_num, output_ptr, Ssub(max));
        // 使用 thrust::transform 计算指数并存储到 output
        thrust::transform(input_ptr, input_ptr + label_num, output_ptr, Sexp());
        
        // 使用 thrust::reduce 计算和并存储结果到 sum
        float sum = thrust::reduce(output_ptr, output_ptr + label_num, 0.0f, thrust::plus<float>());
        
        if (sum > 0.0f) {  // 确保 sum 大于零
            thrust::transform(output_ptr, output_ptr + label_num, output_ptr, Sdiv(sum));
        } else if (sum < 0.0f) {
            // std::cerr << "Error: Sum is negative, cannot proceed with softmax." << std::endl;
            thrust::fill(output_ptr, output_ptr + label_num, 0.0f);  // 将输出设置为0
        } else {
            thrust::fill(output_ptr, output_ptr + label_num, 1.0f / label_num);  // 如果 sum 为零，均匀分配
        }
    }
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