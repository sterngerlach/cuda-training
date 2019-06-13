
/* vector-sum.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

void printCudaError(cudaError_t error, const char* file, const int line)
{
    fprintf(stderr, "Error (%s:%d), code: %d, reason: %s\n",
            file, line, error, cudaGetErrorString(error));

    return;
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

__global__ void sumVectorsOnGPU(float* vecA, float* vecB, float* vecC)
{
    int i = threadIdx.x;

    vecC[i] = vecA[i] + vecB[i];
}

int main(int argc, char** argv)
{
    int dev;
    int numOfElements;
    size_t numOfBytes;

    float* hostVecA;
    float* hostVecB;
    float* hostVecC;
    float* gpuVecA;
    float* gpuVecB;
    float* gpuVecC;
    float* gpuResult;
    
    /* Setup device */
    dev = 0;
    cudaSetDevice(dev);
    
    /* Set vector size */
    numOfElements = 32;
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
    cudaMalloc((float**)&gpuVecA, numOfBytes);
    cudaMalloc((float**)&gpuVecB, numOfBytes);
    cudaMalloc((float**)&gpuVecC, numOfBytes);

    /* Transfer vector data from host to device */
    cudaMemcpy(gpuVecA, hostVecA, numOfBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuVecB, hostVecB, numOfBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuVecC, gpuResult, numOfBytes, cudaMemcpyHostToDevice);

    /* Call CUDA kernel from host */
    dim3 block(numOfElements);
    dim3 grid(1);

    sumVectorsOnGPU<<<grid, block>>>(gpuVecA, gpuVecB, gpuVecC);
    printf("Execution configuration: <<<%d, %d>>>\n", grid.x, block.x);

    /* Copy CUDA kernel result to host */
    cudaMemcpy(gpuResult, gpuVecC, numOfBytes, cudaMemcpyDeviceToHost);

    /* Add vectors in host to check device result */
    sumVectorsOnHost(hostVecA, hostVecB, hostVecC, numOfElements);

    /* Check device result */
    checkResult(hostVecC, gpuResult, numOfElements);

    /* Free device global memory */
    cudaFree(gpuVecA);
    cudaFree(gpuVecB);
    cudaFree(gpuVecC);

    /* Free host memory */
    free(hostVecA);
    free(hostVecB);
    free(hostVecC);
    free(gpuResult);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}

