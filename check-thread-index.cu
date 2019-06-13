
/* check-thread-index.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
    } \

void printMatrix(int* matC, int row, int col)
{
    printf("Matrix (%d, %d)\n", row, col);

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j)
            printf("%2d ", matC[i * col + j]);

        printf("\n");
    }

    printf("\n");

    return;
}

__global__ void printThreadIndex(int* matA, int row, int col)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int i = y * col + x;

    printf("threadIdx: (%d, %d, %d), blockIdx: (%d, %d, %d), "
           "coordinate: (%d, %d), array index: %d, "
           "matrix value: %d\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           x, y, i, matA[i]);
}

int main(int argc, char** argv)
{
    int i;

    int dev;
    cudaDeviceProp deviceProp;

    int matRow;
    int matCol;
    int numOfElements;
    int numOfBytes;

    int* hostMatA;
    int* devMatA;

    /* Setup device */
    dev = 0;
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev));
    
    printf("Using device %d: %s\n", dev, deviceProp.name);

    CHECK_CUDA_CALL(cudaSetDevice(dev));

    /* Set matrix size */
    matCol = 8;
    matRow = 6;
    numOfElements = matRow * matCol;
    numOfBytes = numOfElements * sizeof(int);

    /* Allocate host memory */
    hostMatA = (int*)calloc(numOfElements, sizeof(int));

    for (i = 0; i < numOfElements; ++i)
        hostMatA[i] = i;

    printMatrix(hostMatA, matRow, matCol);

    /* Allocate device memory */
    CHECK_CUDA_CALL(cudaMalloc((void**)&devMatA, numOfBytes));

    /* Set execution configuration */
    dim3 block(4, 2);
    dim3 grid((matCol + block.x - 1) / block.x, (matRow + block.y - 1) / block.y);

    /* Transfer vector data from host */
    CHECK_CUDA_CALL(cudaMemcpy(devMatA, hostMatA, numOfBytes, cudaMemcpyHostToDevice));

    /* Call CUDA kernel from host */
    printThreadIndex<<<grid, block>>>(devMatA, matRow, matCol);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    /* Free device and host memory */
    CHECK_CUDA_CALL(cudaFree(devMatA));
    free(hostMatA);

    /* Reset device */
    CHECK_CUDA_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}

