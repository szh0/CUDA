
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <iostream>
#include <stdlib.h>
#include <cmath>
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
#define N 4
#define K 6
#define M 8

// C(n,m) = A(n,k) * B(k,m)
void gemm_gpu_NN(const float *A, const float *B, float *C, const int n, const int k, const int m) { 
    int lda = k;
    int ldb = m;
    int ldc = m;
    const float alf = 1, bet = 0; 
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
void gemm_gpu_TN(const float *A, const float *B, float *C, const int n, const int k, const int m) { 
    int lda = n;
    int ldb = m;
    int ldc = m;
    const float alf = 1, bet = 0; 
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
void gemm_gpu_NT(const float *A, const float *B, float *C, const int n, const int k, const int m) { 
    int lda = k;
    int ldb = k;
    int ldc = m;
    const float alf = 1, bet = 0; 
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
void gemm_gpu_TT(const float *A, const float *B, float *C, const int n, const int k, const int m) { 
    int lda = n;
    int ldb = k;
    int ldc = m;
    const float alf = 1, bet = 0; 
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

void gemm_gpu_NN_test()
{
    //matrix test 
    const int A_ROW = N;
    const int A_COL = K;
    const int B_ROW = K;
    const int B_COL = M;

    thrust::device_vector<float> d_A(A_ROW * A_COL);
    thrust::device_vector<float> d_B(B_ROW * B_COL);
    thrust::device_vector<float> d_C(A_ROW * B_COL);
    
    matrix_init(thrust::raw_pointer_cast(d_A.data()), A_ROW, A_COL);
    matrix_init(thrust::raw_pointer_cast(d_B.data()), B_ROW, B_COL);

    gemm_gpu_NN(
        thrust::raw_pointer_cast(d_A.data()), 
        thrust::raw_pointer_cast(d_B.data()), 
        thrust::raw_pointer_cast(d_C.data()), 
        A_ROW, 
        A_COL, 
        B_COL
    );

    thrust::host_vector<float> h_A = d_A;
    thrust::host_vector<float> h_B = d_B;
    thrust::host_vector<float> h_C = d_C;

    cudaDeviceSynchronize();

    printf("Matrix A:\n");
    for(int i=0;i< A_ROW * A_COL;++i) {
        printf("%.2f ", h_A[i]);
        if ((i+1) % A_COL == 0) putchar('\n');
    }
    printf("Matrix B:\n");
    for(int i=0;i< B_ROW * B_COL;++i) {
        printf("%.2f ", h_B[i]);
        if ((i+1) % B_COL == 0) putchar('\n');
    }
    printf("Matrix C:\n");
    for(int i=0;i< A_ROW * B_COL;++i) {
        printf("%.2f ", h_C[i]);
        if ((i+1) % B_COL == 0) putchar('\n');
    }
    bool flag = true;
    printf("Real C:\n");
    for(int i = 0; i < A_ROW; i++, putchar('\n'))
        for(int j = 0 ; j < B_COL; j++){
            float res = 0;
            for(int k = 0; k < B_ROW; k++)
                res += h_A[i * A_COL + k] * h_B[k * B_COL +j];

            if(fabs(res - h_C[i * B_COL + j]) > 1e-6)flag = false, printf("%d %d %.2lf %.2lf\n",i,j,res,h_C[i * B_COL + j]);
            printf("%.2lf ",res);
        }
    if(flag) printf("Test passed!\n");
}

void gemm_gpu_TN_test()
{
    //matrix test 
    const int A_ROW = K;
    const int A_COL = N;
    const int B_ROW = K;
    const int B_COL = M;

    thrust::device_vector<float> d_A(A_ROW * A_COL);
    thrust::device_vector<float> d_B(B_ROW * B_COL);
    thrust::device_vector<float> d_C(A_ROW * B_COL);
    
    matrix_init(thrust::raw_pointer_cast(d_A.data()), A_ROW, A_COL);
    matrix_init(thrust::raw_pointer_cast(d_B.data()), B_ROW, B_COL);

    gemm_gpu_TN(
        thrust::raw_pointer_cast(d_A.data()), 
        thrust::raw_pointer_cast(d_B.data()), 
        thrust::raw_pointer_cast(d_C.data()), 
        A_COL, 
        A_ROW, 
        B_COL
    );

    thrust::host_vector<float> h_A = d_A;
    thrust::host_vector<float> h_B = d_B;
    thrust::host_vector<float> h_C = d_C;

    cudaDeviceSynchronize();

    printf("Matrix A:\n");
    for(int i=0;i< A_ROW * A_COL;++i) {
        printf("%.2f ", h_A[i]);
        if ((i+1) % A_COL == 0) putchar('\n');
    }
    printf("Matrix B:\n");
    for(int i=0;i< B_ROW * B_COL;++i) {
        printf("%.2f ", h_B[i]);
        if ((i+1) % B_COL == 0) putchar('\n');
    }
    printf("Matrix C:\n");
    for(int i=0;i< A_COL * B_COL;++i) {
        printf("%.2f ", h_C[i]);
        if ((i+1) % B_COL == 0) putchar('\n');
    }
    bool flag = true;
    printf("Real C:\n");
    for(int i = 0; i < A_COL; i++, putchar('\n'))
        for(int j = 0 ; j < B_COL; j++){
            float res = 0;
            for(int k = 0; k < B_ROW; k++){
                res += h_A[k * A_COL + i] * h_B[k * B_COL +j];
                // printf("%.2lf %.2lf %.2lf\n",h_A[k * A_ROW + i],h_B[k * B_COL +j],res);
            }
            // printf("%.2lf\n",h_A[(B_ROW - 1) * A_COL + i]);
            if(fabs(res - h_C[i * B_COL + j]) > 1e-6)flag = false, printf("%d %d %.2lf %.2lf\n",i,j,res,h_C[i * B_COL + j]);
            else printf("%.2lf ",res);
        }
    if(flag) printf("Test passed!\n");
}

void gemm_gpu_NT_test()
{
    //matrix test 
    const int A_ROW = N;
    const int A_COL = K;
    const int B_ROW = M;
    const int B_COL = K;

    thrust::device_vector<float> d_A(A_ROW * A_COL);
    thrust::device_vector<float> d_B(B_ROW * B_COL);
    thrust::device_vector<float> d_C(A_ROW * B_ROW);
    
    matrix_init(thrust::raw_pointer_cast(d_A.data()), A_ROW, A_COL);
    matrix_init(thrust::raw_pointer_cast(d_B.data()), B_ROW, B_COL);

    gemm_gpu_NT(
        thrust::raw_pointer_cast(d_A.data()), 
        thrust::raw_pointer_cast(d_B.data()), 
        thrust::raw_pointer_cast(d_C.data()), 
        A_ROW, 
        A_COL, 
        B_ROW
    );

    thrust::host_vector<float> h_A = d_A;
    thrust::host_vector<float> h_B = d_B;
    thrust::host_vector<float> h_C = d_C;

    cudaDeviceSynchronize();

    printf("Matrix A:\n");
    for(int i=0;i< A_ROW * A_COL;++i) {
        printf("%.2f ", h_A[i]);
        if ((i+1) % A_COL == 0) putchar('\n');
    }
    printf("Matrix B:\n");
    for(int i=0;i< B_ROW * B_COL;++i) {
        printf("%.2f ", h_B[i]);
        if ((i+1) % B_COL == 0) putchar('\n');
    }
    printf("Matrix C:\n");
    for(int i=0;i< A_ROW * B_ROW;++i) {
        printf("%.2f ", h_C[i]);
        if ((i+1) % B_ROW == 0) putchar('\n');
    }
    bool flag = true;
    printf("Real C:\n");
    for(int i = 0; i < A_ROW; i++, putchar('\n'))
        for(int j = 0 ; j < B_ROW; j++){
            float res = 0;
            for(int k = 0; k < B_COL; k++){
                res += h_A[i * A_COL + k] * h_B[j * B_COL + k];
                // printf("%.2lf %.2lf %.2lf\n",h_A[k * A_ROW + i],h_B[k * B_COL +j],res);
            }
            // printf("%.2lf\n",h_A[(B_ROW - 1) * A_COL + i]);
            if(fabs(res - h_C[i * B_ROW + j]) > 1e-6)flag = false, printf("%d %d %.2lf %.2lf\n",i,j,res,h_C[i * B_ROW+ j]);
            else printf("%.2lf ",res);
        }
    if(flag) printf("Test passed!\n");
}

void gemm_gpu_TT_test()
{
    //matrix test 
    const int A_ROW = K;
    const int A_COL = N;
    const int B_ROW = M;
    const int B_COL = K;

    thrust::device_vector<float> d_A(A_ROW * A_COL);
    thrust::device_vector<float> d_B(B_ROW * B_COL);
    thrust::device_vector<float> d_C(A_COL * B_ROW);
    
    matrix_init(thrust::raw_pointer_cast(d_A.data()), A_ROW, A_COL);
    matrix_init(thrust::raw_pointer_cast(d_B.data()), B_ROW, B_COL);

    gemm_gpu_TT(
        thrust::raw_pointer_cast(d_A.data()), 
        thrust::raw_pointer_cast(d_B.data()), 
        thrust::raw_pointer_cast(d_C.data()), 
        A_COL, 
        A_ROW, 
        B_ROW
    );

    thrust::host_vector<float> h_A = d_A;
    thrust::host_vector<float> h_B = d_B;
    thrust::host_vector<float> h_C = d_C;

    cudaDeviceSynchronize();

    printf("Matrix A:\n");
    for(int i=0;i< A_ROW * A_COL;++i) {
        printf("%.2f ", h_A[i]);
        if ((i+1) % A_COL == 0) putchar('\n');
    }
    printf("Matrix B:\n");
    for(int i=0;i< B_ROW * B_COL;++i) {
        printf("%.2f ", h_B[i]);
        if ((i+1) % B_COL == 0) putchar('\n');
    }
    printf("Matrix C:\n");
    for(int i=0;i< A_COL* B_ROW;++i) {
        printf("%.2f ", h_C[i]);
        if ((i+1) % B_ROW == 0) putchar('\n');
    }
    bool flag = true;
    printf("Real C:\n");
    for(int i = 0; i < A_COL; i++, putchar('\n'))
        for(int j = 0 ; j < B_ROW; j++){
            float res = 0;
            for(int k = 0; k < B_COL; k++){
                res += h_A[k * A_COL + i] * h_B[j * B_COL + k];
                // printf("%.2lf %.2lf %.2lf\n",h_A[k * A_ROW + i],h_B[k * B_COL +j],res);
            }
            // printf("%.2lf\n",h_A[(B_ROW - 1) * A_COL + i]);
            if(fabs(res - h_C[i * B_ROW + j]) > 1e-6)flag = false, printf("%d %d %.2lf %.2lf\n",i,j,res,h_C[i * B_ROW+ j]);
            else printf("%.2lf ",res);
        }
    if(flag) printf("Test passed!\n");
}

int main(){

    gemm_gpu_NN_test();
    gemm_gpu_TN_test();
    gemm_gpu_NT_test();
    gemm_gpu_TT_test();
    return 0;
}