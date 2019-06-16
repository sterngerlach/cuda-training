
/* matrix-sum.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/*
 * Check kernel performance using the following commands
 * 
 * nvcc -O3 matrix-sum.cu -o bin/matrix-sum
 * 
 * su
 * nvprof --metrics achieved_occupancy bin/matrix-sum 32 32
 * nvprof --metrics gld_throughput bin/matrix-sum 32 32
 * nvprof --metrics gld_efficiency bin/matrix-sum 32 32
 */

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

void initializeMatrix(float* matC, int row, int col)
{
    int i;
    int j;
    float* pC = matC;

    srand((unsigned int)time(NULL));

    for (i = 0; i < row; ++i) {
        for (j = 0; j < col; ++j)
            pC[j] = (float)(rand() & 0xFF) / 10.0f;

        pC += col;
    }

    return;
}

__global__ void sumMatrixOnGPU2D(float* matA, float* matB, float* matC,
                                 int row, int col)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int i = y * col + x;

    if (x < col && y < row)
        matC[i] = matA[i] + matB[i];
}

int main(int argc, char** argv)
{
    int dev;
    cudaDeviceProp deviceProp;

    int matRow;
    int matCol;
    int numOfElements;
    int numOfBytes;

    int blockRow;
    int blockCol;
    
    /*
    float* hostMatA;
    float* hostMatB;
    float* devResult;
    */

    float* devMatA;
    float* devMatB;
    float* devMatC;

    struct timeval startTime;
    struct timeval endTime;

    /* Setup device */
    dev = 0;
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev));
    
    printf("Using device %d: %s\n", dev, deviceProp.name);

    CHECK_CUDA_CALL(cudaSetDevice(dev));

    /* Set matrix size */
    matCol = 1 << 14;
    matRow = 1 << 14;
    numOfElements = matRow * matCol;
    numOfBytes = numOfElements * sizeof(float);

    /* Allocate host memory */
    /*
    hostMatA = (float*)calloc(numOfElements, sizeof(float));
    hostMatB = (float*)calloc(numOfElements, sizeof(float));
    devResult = (float*)calloc(numOfElements, sizeof(float));
    */

    /*
    initializeMatrix(hostMatA, matRow, matCol);
    initializeMatrix(hostMatB, matRow, matCol);
    */

    /* Allocate device memory */
    CHECK_CUDA_CALL(cudaMalloc((void**)&devMatA, numOfBytes));
    CHECK_CUDA_CALL(cudaMalloc((void**)&devMatB, numOfBytes));
    CHECK_CUDA_CALL(cudaMalloc((void**)&devMatC, numOfBytes));

    /* Transfer matrix data from host */
    /*
    CHECK_CUDA_CALL(cudaMemcpy(devMatA, hostMatA, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(devMatB, hostMatB, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(devMatC, devResult, numOfBytes, cudaMemcpyHostToDevice));
    */

    /* Set execution configuration */
    blockRow = 32;
    blockCol = 32;

    if (argc > 1)
        blockCol = atoi(argv[1]);

    if (argc > 2)
        blockRow = atoi(argv[2]);

    dim3 block(blockCol, blockRow);
    dim3 grid((matCol + block.x - 1) / block.x, (matRow + block.y - 1) / block.y);
    
    /* Call CUDA kernel from host */
    gettimeofday(&startTime, NULL);
    sumMatrixOnGPU2D<<<grid, block>>>(devMatA, devMatB, devMatC, matRow, matCol);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);

    printf("Execution configuration: <<<(%d, %d), (%d, %d)>>>\n",
           grid.x, grid.y, block.x, block.y);
    printf("Device execution time: %.6f\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6));

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());

    /* Free device and host memory */
    CHECK_CUDA_CALL(cudaFree(devMatA));
    CHECK_CUDA_CALL(cudaFree(devMatB));
    CHECK_CUDA_CALL(cudaFree(devMatC));
    
    /*
    free(hostMatA);
    free(hostMatB);
    free(devResult);
    */

    /* Reset device */
    CHECK_CUDA_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}

