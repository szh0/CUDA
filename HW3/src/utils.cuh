// gemm and other utils functions
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
// #include <device_launch_parameters_h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

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
void curand_init(float *A, int total_size) { 
    // Create a pseudo-random number generator 
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT); 
    
    // Set the seed for the random number generator using the system clock 
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock()); 
    
    // Fill the array with random numbers on the device 
    curandGenerateUniform(prng, A, total_size); 
    curandDestroyGenerator(prng); 
}

void tensor_init(Tensor &tensor) {
    assert(tensor.device == "gpu");
    curand_init(tensor.data, tensor.size());
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