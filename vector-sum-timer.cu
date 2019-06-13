
/* vector-sum-timer.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

void printCudaError(cudaError_t error, const char* file, const int line)
{
    fprintf(stderr, "Error (%s:%d), code: %d, reason: %s\n",
            file, line, error, cudaGetErrorString(error));

    return;
}

void printCudaErrorAndExit(cudaError_t error, const char* file, const int line)
{
    printCudaError(error, file, line);
    exit(EXIT_FAILURE);
}

void checkResult(float* hostResult, float* devResult, int vecSize)
{
    const double epsilon = 1.0E-8;

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
    cudaError_t error;

    int numOfElements;
    size_t numOfBytes;
    
    struct timeval startTime;
    struct timeval endTime;

    float* hostVecA;
    float* hostVecB;
    float* hostVecC;
    float* gpuVecA;
    float* gpuVecB;
    float* gpuVecC;
    float* gpuResult;
    
    /* Setup device */
    dev = 0;

    if ((error = cudaGetDeviceProperties(&deviceProp, dev)) != cudaSuccess)
        printCudaErrorAndExit(error, __FILE__, __LINE__);
    
    printf("Using device %d: %s\n", dev, deviceProp.name);

    if ((error = cudaSetDevice(dev)) != cudaSuccess)
        printCudaErrorAndExit(error, __FILE__, __LINE__);
    
    /* Set vector size */
    numOfElements = 1 << 24;
    numOfBytes = numOfElements * sizeof(float);
    printf("Vector size: %d\n", numOfElements);
    
    /* Allocate host memory */
    hostVecA = (float*)calloc(numOfElements, sizeof(float));
    hostVecB = (float*)calloc(numOfElements, sizeof(float));
    hostVecC = (float*)calloc(numOfElements, sizeof(float));
    gpuResult = (float*)calloc(numOfElements, sizeof(float));
    
    /* Initialize vectors */
    initializeVector(hostVecA, numOfElements);
    initializeVector(hostVecB, numOfElements);
    
    /* Allocate device memory */
    if ((error = cudaMalloc((float**)&gpuVecA, numOfBytes)) != cudaSuccess)
        printCudaErrorAndExit(error, __FILE__, __LINE__);

    if ((error = cudaMalloc((float**)&gpuVecB, numOfBytes)) != cudaSuccess)
        printCudaErrorAndExit(error, __FILE__, __LINE__);

    if ((error = cudaMalloc((float**)&gpuVecC, numOfBytes)) != cudaSuccess)
        printCudaErrorAndExit(error, __FILE__, __LINE__);

    /* Transfer vector data from host to device */
    if ((error = cudaMemcpy(gpuVecA, hostVecA,
                            numOfBytes, cudaMemcpyHostToDevice)) != cudaSuccess)
        printCudaErrorAndExit(error, __FILE__, __LINE__);

    if ((error = cudaMemcpy(gpuVecB, hostVecB,
                            numOfBytes, cudaMemcpyHostToDevice)) != cudaSuccess)
        printCudaErrorAndExit(error, __FILE__, __LINE__);

    if ((error = cudaMemcpy(gpuVecC, gpuResult,
                            numOfBytes, cudaMemcpyHostToDevice)) != cudaSuccess)
        printCudaErrorAndExit(error, __FILE__, __LINE__);

    /* Call CUDA kernel from host */
    dim3 block(1024);
    dim3 grid((numOfElements + block.x - 1) / block.x);
    
    gettimeofday(&startTime, NULL);
    sumVectorsOnGPU<<<grid, block>>>(gpuVecA, gpuVecB, gpuVecC, numOfElements);
    
    if ((error = cudaDeviceSynchronize()) != cudaSuccess)
        printCudaErrorAndExit(error, __FILE__, __LINE__);

    gettimeofday(&endTime, NULL);

    printf("Execution configuration: <<<%d, %d>>>\n", grid.x, block.x);
    printf("GPU execution time: %.4f\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6));

    /* Check kernel error */
    if ((error = cudaGetLastError()) != cudaSuccess)
        printCudaErrorAndExit(error, __FILE__, __LINE__);

    /* Copy CUDA kernel result to host */
    if ((error = cudaMemcpy(gpuResult, gpuVecC,
                            numOfBytes, cudaMemcpyDeviceToHost)) != cudaSuccess)
        printCudaErrorAndExit(error, __FILE__, __LINE__);

    /* Add vectors in host to check device result */
    gettimeofday(&startTime, NULL);
    sumVectorsOnHost(hostVecA, hostVecB, hostVecC, numOfElements);
    gettimeofday(&endTime, NULL);

    printf("Host execution time: %.4f\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6));

    /* Check device result */
    checkResult(hostVecC, gpuResult, numOfElements);

    /* Free device global memory */
    if ((error = cudaFree(gpuVecA)))
        printCudaErrorAndExit(error, __FILE__, __LINE__);

    if ((error = cudaFree(gpuVecB)))
        printCudaErrorAndExit(error, __FILE__, __LINE__);

    if ((error = cudaFree(gpuVecC)))
        printCudaErrorAndExit(error, __FILE__, __LINE__);

    /* Free host memory */
    free(hostVecA);
    free(hostVecB);
    free(hostVecC);
    free(gpuResult);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}

