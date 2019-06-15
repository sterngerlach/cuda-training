
/* simple-warp-divergence.cu */

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

__global__ void warmUp(float* c)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    c[id] = 0.0f;
}

__global__ void warpDivergence(float* c)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.0f;
    float b = 0.0f;

    if (id % 2 == 0)
        a = 100.0f;
    else
        b = 200.0f;

    c[id] = a + b;
}

__global__ void noWarpDivergence(float* c)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.0f;
    float b = 0.0f;

    if ((id / warpSize) % 2 == 0)
        a = 100.0f;
    else
        b = 200.0f;

    c[id] = a + b;
}

__global__ void warpDivergencePredicate(float* c)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.0f;
    float b = 0.0f;
    bool pred = (id % 2 == 0);

    if (pred)
        a = 100.0f;

    if (!pred)
        b = 200.0f;

    c[id] = a + b;
}

int main(int argc, char** argv)
{
    int dev;
    cudaDeviceProp deviceProp;

    int size;
    int blockSize;
    
    size_t numOfBytes;
    float* devC;

    struct timeval startTime;
    struct timeval endTime;
    
    /* Setup device */
    dev = 0;
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Using device %d: %s\n", dev, deviceProp.name);

    /* Set data size */
    if (argc > 1)
        blockSize = atoi(argv[1]);
    else
        blockSize = 64;

    if (argc > 2)
        size = atoi(argv[2]);
    else
        size = 64;
    
    printf("Data size: %d, Block size: %d\n", size, blockSize);

    /* Set execution configuration */
    dim3 block(blockSize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);

    printf("Execution configuration: <<<(%d, %d), (%d, %d)>>>\n",
           grid.x, grid.y, block.x, block.y);

    numOfBytes = size * sizeof(float);
    CHECK_CUDA_CALL(cudaMalloc((float**)&devC, numOfBytes));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    /* Call kernel for warming up */
    gettimeofday(&startTime, NULL);
    warmUp<<<grid, block>>>(devC);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);

    printf("Warmup execution time: %.6f\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6));

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());

    /* Call kernel that causes warp divergence */
    gettimeofday(&startTime, NULL);
    warpDivergence<<<grid, block>>>(devC);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);
    
    printf("WarpDivergence execution time: %.6f\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6));

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());

    /* Call kernel that does not cause warp divergence */
    gettimeofday(&startTime, NULL);
    noWarpDivergence<<<grid, block>>>(devC);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);
    
    printf("NoWarpDivergence execution time: %.6f\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6));

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());

    /* Call kernel that uses predicates */
    gettimeofday(&startTime, NULL);
    warpDivergencePredicate<<<grid, block>>>(devC);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);
    
    printf("WarpDivergencePredicate execution time: %.6f\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6));

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());

    /* Free device memory */
    CHECK_CUDA_CALL(cudaFree(devC));

    /* Reset device */
    CHECK_CUDA_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}

