
/* vector-sum-timer.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_CALL(call) \
    { \
        const cudaError_t error = call; \
        \
        if (error != cudaSuccess) { \
            fprintf(stderr, "Error (%s:%d), code: %d, reason: %s\n", \
                    __FILE__, __LINE__, \
                    error, cudaGetErrorString(error)); \
                exit(EXIT_FAILURE); \
        } \
    }

void checkResult(float* hostResult, float* devResult, int vecSize)
{
    const double epsilon = 1.0e-8;

    int i;
    
    for (i = 0; i < vecSize; ++i) {
        if (fabsf(hostResult[i] - devResult[i]) > epsilon) {
            fprintf(stderr, "Results do not match!\n");
            fprintf(stderr, "Index %d, Host result: %5.2f, GPU result: %5.2f",
                    i, hostResult[i], devResult[i]);
            return;
        }
    }

    printf("Results match.\n");

    return;
}

void initializeVector(float* vec, int vecSize)
{
    int i;

    srand((unsigned int)time(NULL));

    for (i = 0; i < vecSize; ++i)
        vec[i] = (float)(rand() & 0xFF) / 10.0f;

    return;
}

void sumVectorsOnHost(float* vecA, float* vecB, float* vecC, int vecSize)
{
    int i;

    for (i = 0; i < vecSize; ++i)
        vecC[i] = vecA[i] + vecB[i];

    return;
}

__global__ void sumVectorsOnGPU(float* vecA, float* vecB, float* vecC, int vecSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < vecSize)
        vecC[i] = vecA[i] + vecB[i];
}

int main(int argc, char** argv)
{
    int dev;
    cudaDeviceProp deviceProp;

    int numOfElements;
    size_t numOfBytes;
    
    float* hostVecA;
    float* hostVecB;
    float* hostVecC;
    float* devVecA;
    float* devVecB;
    float* devVecC;
    float* devResult;

    struct timeval startTime;
    struct timeval endTime;

    /* Setup device */
    dev = 0;

    CHECK_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);

    CHECK_CUDA_CALL(cudaSetDevice(dev));
    
    /* Set vector size */
    numOfElements = 1 << 24;
    numOfBytes = numOfElements * sizeof(float);
    printf("Vector size: %d\n", numOfElements);
    
    /* Allocate host memory */
    hostVecA = (float*)calloc(numOfElements, sizeof(float));
    hostVecB = (float*)calloc(numOfElements, sizeof(float));
    hostVecC = (float*)calloc(numOfElements, sizeof(float));
    devResult = (float*)calloc(numOfElements, sizeof(float));
    
    /* Initialize vectors */
    initializeVector(hostVecA, numOfElements);
    initializeVector(hostVecB, numOfElements);
    
    /* Allocate device memory */
    CHECK_CUDA_CALL(cudaMalloc((float**)&devVecA, numOfBytes));
    CHECK_CUDA_CALL(cudaMalloc((float**)&devVecB, numOfBytes));
    CHECK_CUDA_CALL(cudaMalloc((float**)&devVecC, numOfBytes));

    /* Transfer vector data from host to device */
    CHECK_CUDA_CALL(cudaMemcpy(devVecA, hostVecA, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(devVecB, hostVecB, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(devVecC, devResult, numOfBytes, cudaMemcpyHostToDevice));

    /* Call kernel from host */
    dim3 block(1024);
    dim3 grid((numOfElements + block.x - 1) / block.x);
    
    gettimeofday(&startTime, NULL);
    sumVectorsOnGPU<<<grid, block>>>(devVecA, devVecB, devVecC, numOfElements);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);

    printf("Execution configuration: <<<%d, %d>>>\n", grid.x, block.x);
    printf("GPU execution time: %.6f\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6));

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());

    /* Copy CUDA kernel result to host */
    CHECK_CUDA_CALL(cudaMemcpy(devResult, devVecC, numOfBytes, cudaMemcpyDeviceToHost));

    /* Add vectors in host to check device result */
    gettimeofday(&startTime, NULL);
    sumVectorsOnHost(hostVecA, hostVecB, hostVecC, numOfElements);
    gettimeofday(&endTime, NULL);

    printf("Host execution time: %.6f\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6));

    /* Check device result */
    checkResult(hostVecC, devResult, numOfElements);

    /* Free device global memory */
    CHECK_CUDA_CALL(cudaFree(devVecA));
    CHECK_CUDA_CALL(cudaFree(devVecB));
    CHECK_CUDA_CALL(cudaFree(devVecC));

    /* Free host memory */
    free(hostVecA);
    free(hostVecB);
    free(hostVecC);
    free(devResult);

    /* Reset device */
    CHECK_CUDA_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}

