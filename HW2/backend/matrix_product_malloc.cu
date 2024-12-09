
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <iostream>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <curand.h>


#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int kCudaThreadsNum = 512;//threads in block
inline int CudaGetBlocks(int N) { 
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum; 
}

// C(n,m) = A(n,k) * B(k,m) 
// no bias
void gemm_gpu(const float *A, const float *B, float *C, const int n, const int k, const int m) { 
    int A_ROW = n, A_COL = k;
    int B_ROW = k, B_COL = m;
    const float alf = 1, bet = 0; 
    const float *alpha = &alf; 
    const float *beta = &bet; 
    // Create a handle for CUBLAS 
    cublasHandle_t handle; 
    cublasCreate(&handle); 
    // Do the actual multiplication 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_COL, A_ROW, B_ROW, alpha, B, B_COL, A, A_COL, beta, C, B_COL); 
    // Destroy the handle cublasDestroy(handle); 
    cublasDestroy(handle);
}

// C(n,m) = A(n,k) * B(k,m) + bias(n,m)
void gemm_gpu_bias(const float *A, const float *B, float *C, float* beta, const int n, const int k, const int m) { 
    int A_ROW = n, A_COL = k;
    int B_ROW = k, B_COL = m;
    const float alf = 1; 
    const float *alpha = &alf; 
    // Create a handle for CUBLAS 
    cublasHandle_t handle; 
    cublasCreate(&handle); 
    // Do the actual multiplication 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_COL, A_ROW, B_ROW, alpha, B, B_COL, A, A_COL, beta, C, B_COL); 
    // Destroy the handle cublasDestroy(handle); 
    cublasDestroy(handle);
}

// C(n,m) = A^T(n,k) * B(k,m) + bias(n,m)
// A(k,n)
void gemm_gpu_AT_bias(const float *A, const float *B, float *C, float* beta, const int n, const int k, const int m) { 
    int A_ROW = n, A_COL = k;
    int B_ROW = k, B_COL = m;
    const float alf = 1; 
    const float *alpha = &alf; 
    // Create a handle for CUBLAS 
    cublasHandle_t handle; 
    cublasCreate(&handle); 
    // Do the actual multiplication 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_COL, A_ROW, B_ROW, alpha, B, B_COL, A, A_COL, beta, C, B_COL); 
    // Destroy the handle cublasDestroy(handle); 
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


int main(){

    //matrix test 
    const int A_ROW = 5;
    const int A_COL = 6;
    const int B_ROW = 6;
    const int B_COL = 7;
    float *d_A = nullptr, *d_B = nullptr, *d_C= nullptr;

    cudaMalloc(&d_A, sizeof(float) * A_ROW * A_COL); //在显存中开辟空间
    cudaMalloc(&d_B, sizeof(float) * B_ROW * B_COL);
    cudaMalloc(&d_C, sizeof(float) * A_ROW * B_COL);    

    cudaDeviceSynchronize();
    
    matrix_init(d_A, A_ROW, A_COL);
    matrix_init(d_B, B_ROW, B_COL);

    cudaDeviceSynchronize();

    gemm_gpu(d_A, d_B, d_C, A_ROW, A_COL, B_COL);

    float *h_A, *h_B, *h_C;

    h_A = (float*)malloc(sizeof(float) * A_ROW * A_COL);  //在内存中开辟空间 
    h_B = (float*)malloc(sizeof(float) * B_ROW * B_COL);
    h_C = (float*)malloc(sizeof(float) * A_ROW * B_COL);

    cudaDeviceSynchronize();
    
    cudaMemcpy(h_A, d_A, sizeof(float) * A_ROW * A_COL, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, sizeof(float) * B_ROW * B_COL, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, sizeof(float) * A_ROW * B_COL, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();


    //print matrix
    printf("Matrix A:\n");
    for(int i=0;i< A_ROW * A_COL;++i) {
        printf("%.2f ",h_A[i]);
        if ((i+1) % A_COL == 0) putchar('\n');
    }
    printf("Matrix B:\n");
    for(int i=0;i< B_ROW * B_COL;++i) {
        printf("%.2f ",h_B[i]);
        if ((i+1) % B_COL == 0) putchar('\n');
    }
    printf("Matrix C:\n");
    for(int i=0;i< A_ROW * B_COL;++i) {
        printf("%.2f ",h_C[i]);
        if ((i+1) % B_COL == 0) putchar('\n');
    }
    printf("Real C:\n");
    for(int i = 0; i < A_ROW; i++, putchar('\n'))
        for(int j = 0 ; j < B_COL; j++){
            float res = 0;
            for(int k = 0; k < B_ROW; k++)
                res += h_A[i * A_COL + k] * h_B[k * B_COL +j];
            printf("%.2lf ",res);
        }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}