
/* reduce-integer.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/*
 * Check kernel performance using the following commands
 *
 * nvcc reduce-integer.cu -o bin/reduce-integer
 *
 * su
 * nvprof --metrics inst_per_warp bin/reduce-integer
 * nvprof --metrics gld_throughput bin/reduce-integer
 * nvprof --metrics gld_efficiency bin/reduce-integer
 * nvprof --metrics gst_efficiency bin/reduce-integer
 * nvprof --metrics dram_read_throughput bin/reduce-integer
 * nvprof --metrics stall_sync bin/reduce-integer
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

int recursiveReduceHost(int* inArray, size_t size)
{
    int i;
    int stride;

    if (size == 1)
        return inArray[0];

    stride = (int)(size / 2);

    /* Execute in-place reduction */
    for (i = 0; i < stride; ++i)
        inArray[i] += inArray[i + stride];

    /* Recursive call */
    return recursiveReduceHost(inArray, stride);
}

__global__ void reduceNeighbored(int* inArray, int* outArray, unsigned int size)
{
    unsigned int id = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    int* pArray = inArray + blockIdx.x * blockDim.x;
    int stride;

    if (i >= size)
        return;

    for (stride = 1; stride < blockDim.x; stride *= 2) {
        if ((id % (2 * stride)) == 0)
            pArray[id] += pArray[id + stride];
        
        /* Synchronize between all threads in the thread block */
        __syncthreads();
    }

    if (id == 0)
        outArray[blockIdx.x] = pArray[0];
}

__global__ void reduceNeighboredLess(int* inArray, int* outArray, unsigned int size)
{
    unsigned int id = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j;
    int* pArray = inArray + blockIdx.x * blockDim.x;
    int stride;

    if (i >= size)
        return;

    for (stride = 1; stride < blockDim.x; stride *= 2) {
        /* Calculate local array index from the thread index */
        /* This prevents warp divergences to occur */
        j = 2 * stride * id;

        if (j < blockDim.x)
            pArray[j] += pArray[j + stride];

        /* Synchronize between all threads in the thread block */
        __syncthreads();
    }

    if (id == 0)
        outArray[blockIdx.x] = pArray[0];
}

__global__ void reduceInterleaved(int* inArray, int* outArray, unsigned int size)
{
    unsigned int id = threadIdx.x;
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    int* pArray = inArray + blockIdx.x * blockDim.x;
    int stride;

    if (i >= size)
        return;

    for (stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (id < stride)
            pArray[id] += pArray[id + stride];

        /* Synchronize between all threads in the thread block */
        __syncthreads();
    }

    if (id == 0)
        outArray[blockIdx.x] = pArray[0];
}

__global__ void reduceUnrolling2(int* inArray, int* outArray, unsigned int size)
{
    unsigned int id = threadIdx.x;
    unsigned int i = threadIdx.x + (blockIdx.x * 2) * blockDim.x;
    int* pArray = inArray + (blockIdx.x * 2) * blockDim.x;
    int stride;
    
    /* Each thread processes 2 data blocks (cyclic) */
    if (i + blockDim.x < size)
        inArray[i] += inArray[i + blockDim.x];

    __syncthreads();

    for (stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (id < stride)
            pArray[id] += pArray[id + stride];

        /* Synchronize between all threads in the thread block */
        __syncthreads();
    }

    if (id == 0)
        outArray[blockIdx.x] = pArray[0];
}

__global__ void reduceUnrolling4(int* inArray, int* outArray, unsigned int size)
{
    unsigned int id = threadIdx.x;
    unsigned int i = threadIdx.x + (blockIdx.x * 4) * blockDim.x;
    int* pArray = inArray + (blockIdx.x * 4) * blockDim.x;
    int stride;
    
    /* Each thread processes 4 data blocks (cyclic) */
    if (i + blockDim.x * 3 < size) {
        inArray[i] += inArray[i + blockDim.x];
        inArray[i] += inArray[i + blockDim.x * 2];
        inArray[i] += inArray[i + blockDim.x * 3];
    }

    __syncthreads();

    for (stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (id < stride)
            pArray[id] += pArray[id + stride];

        /* Synchronize between all threads in the thread block */
        __syncthreads();
    }

    if (id == 0)
        outArray[blockIdx.x] = pArray[0];
}

__global__ void reduceUnrolling8(int* inArray, int* outArray, unsigned int size)
{
    unsigned int id = threadIdx.x;
    unsigned int i = threadIdx.x + (blockIdx.x * 8) * blockDim.x;
    int* pArray = inArray + (blockIdx.x * 8) * blockDim.x;
    int stride;
    
    /* Each thread processes 8 data blocks (cyclic) */
    if (i + blockDim.x * 7 < size) {
        inArray[i] += inArray[i + blockDim.x];
        inArray[i] += inArray[i + blockDim.x * 2];
        inArray[i] += inArray[i + blockDim.x * 3];
        inArray[i] += inArray[i + blockDim.x * 4];
        inArray[i] += inArray[i + blockDim.x * 5];
        inArray[i] += inArray[i + blockDim.x * 6];
        inArray[i] += inArray[i + blockDim.x * 7];
    }

    __syncthreads();

    for (stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (id < stride)
            pArray[id] += pArray[id + stride];

        /* Synchronize between all threads in the thread block */
        __syncthreads();
    }

    if (id == 0)
        outArray[blockIdx.x] = pArray[0];
}

__global__ void reduceUnrollingWarps8(int* inArray, int* outArray, unsigned int size)
{
    unsigned int id = threadIdx.x;
    unsigned int i = threadIdx.x + (blockIdx.x * 8) * blockDim.x;
    int* pArray = inArray + (blockIdx.x * 8) * blockDim.x;
    volatile int* pvArray;
    int stride;
    
    /* Each thread processes 8 data blocks (cyclic) */
    if (i + blockDim.x * 7 < size) {
        inArray[i] += inArray[i + blockDim.x];
        inArray[i] += inArray[i + blockDim.x * 2];
        inArray[i] += inArray[i + blockDim.x * 3];
        inArray[i] += inArray[i + blockDim.x * 4];
        inArray[i] += inArray[i + blockDim.x * 5];
        inArray[i] += inArray[i + blockDim.x * 6];
        inArray[i] += inArray[i + blockDim.x * 7];
    }

    __syncthreads();

    for (stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (id < stride)
            pArray[id] += pArray[id + stride];

        /* Synchronize between all threads in the thread block */
        __syncthreads();
    }

    /* Reduction using warp unrolling */
    /* Unnecessary __syncthreads() calls are eliminated */
    if (id < 32) {
        pvArray = pArray;
        pvArray[id] += pvArray[id + 32];
        pvArray[id] += pvArray[id + 16];
        pvArray[id] += pvArray[id + 8];
        pvArray[id] += pvArray[id + 4];
        pvArray[id] += pvArray[id + 2];
        pvArray[id] += pvArray[id + 1];
    }

    if (id == 0)
        outArray[blockIdx.x] = pArray[0];
}

__global__ void reduceCompleteUnrollingWarps8(int* inArray, int* outArray, unsigned int size)
{
    unsigned int id = threadIdx.x;
    unsigned int i = threadIdx.x + (blockIdx.x * 8) * blockDim.x;
    int* pArray = inArray + (blockIdx.x * 8) * blockDim.x;
    volatile int* pvArray;
    
    /* Each thread processes 8 data blocks (cyclic) */
    if (i + blockDim.x * 7 < size) {
        inArray[i] += inArray[i + blockDim.x];
        inArray[i] += inArray[i + blockDim.x * 2];
        inArray[i] += inArray[i + blockDim.x * 3];
        inArray[i] += inArray[i + blockDim.x * 4];
        inArray[i] += inArray[i + blockDim.x * 5];
        inArray[i] += inArray[i + blockDim.x * 6];
        inArray[i] += inArray[i + blockDim.x * 7];
    }

    __syncthreads();

    /* In-place reduction and complete unrolling */
    /* Maximum number of the threads in the thread block is 1024 */
    if (blockDim.x >= 1024 && id < 512)
        pArray[id] += pArray[id + 512];

    __syncthreads();

    if (blockDim.x >= 512 && id < 256)
        pArray[id] += pArray[id + 256];

    __syncthreads();

    if (blockDim.x >= 256 && id < 128)
        pArray[id] += pArray[id + 128];

    __syncthreads();

    if (blockDim.x >= 128 && id < 64)
        pArray[id] += pArray[id + 64];

    __syncthreads();

    /* Warp unrolling */
    /* Unnecessary __syncthreads() calls are eliminated */
    if (id < 32) {
        pvArray = pArray;
        pvArray[id] += pvArray[id + 32];
        pvArray[id] += pvArray[id + 16];
        pvArray[id] += pvArray[id + 8];
        pvArray[id] += pvArray[id + 4];
        pvArray[id] += pvArray[id + 2];
        pvArray[id] += pvArray[id + 1];
    }
    
    if (id == 0)
        outArray[blockIdx.x] = pArray[0];
}

template <unsigned int BlockSize>
__global__ void reduceCompleteUnrollingWarps(int* inArray, int* outArray, unsigned int size)
{
    unsigned int id = threadIdx.x;
    unsigned int i = threadIdx.x + (blockIdx.x * 8) * blockDim.x;
    int* pArray = inArray + (blockIdx.x * 8) * blockDim.x;
    volatile int* pvArray;
    
    /* Each thread processes 8 data blocks (cyclic) */
    if (i + blockDim.x * 7 < size) {
        inArray[i] += inArray[i + blockDim.x];
        inArray[i] += inArray[i + blockDim.x * 2];
        inArray[i] += inArray[i + blockDim.x * 3];
        inArray[i] += inArray[i + blockDim.x * 4];
        inArray[i] += inArray[i + blockDim.x * 5];
        inArray[i] += inArray[i + blockDim.x * 6];
        inArray[i] += inArray[i + blockDim.x * 7];
    }

    __syncthreads();
    
    /* In-place reduction and complete unrolling */
    /* Maximum number of the threads in the thread block is 1024 */
    if (BlockSize >= 1024 && id < 512)
        pArray[id] += pArray[id + 512];

    __syncthreads();

    if (BlockSize >= 512 && id < 256)
        pArray[id] += pArray[id + 256];

    __syncthreads();

    if (BlockSize >= 256 && id < 128)
        pArray[id] += pArray[id + 128];

    __syncthreads();

    if (BlockSize >= 128 && id < 64)
        pArray[id] += pArray[id + 64];

    __syncthreads();

    /* Warp unrolling */
    /* Unnecessary __syncthreads() calls are eliminated */
    if (id < 32) {
        pvArray = pArray;
        pvArray[id] += pvArray[id + 32];
        pvArray[id] += pvArray[id + 16];
        pvArray[id] += pvArray[id + 8];
        pvArray[id] += pvArray[id + 4];
        pvArray[id] += pvArray[id + 2];
        pvArray[id] += pvArray[id + 1];
    }
    
    if (id == 0)
        outArray[blockIdx.x] = pArray[0];
}

int main(int argc, char** argv)
{
    int i;

    int dev;
    cudaDeviceProp deviceProp;

    int numOfElements;
    size_t numOfBytes;
    int blockSize;

    int* hostInput;
    int* hostTmp;
    int* hostResult;

    int* devInput;
    int* devResult;

    int hostSum;
    int devSum;

    struct timeval startTime;
    struct timeval endTime;
    
    /* Setup device */
    dev = 0;
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev));
    
    printf("Using device %d: %s\n", dev, deviceProp.name);

    CHECK_CUDA_CALL(cudaSetDevice(dev));
    
    /* Set array size */
    numOfElements = 1 << 24;
    numOfBytes = numOfElements * sizeof(int);
    printf("Array size: %d\n", numOfElements);
    
    /* Set execution configuration */
    blockSize = 512;

    if (argc > 1)
        blockSize = atoi(argv[1]);

    dim3 block(blockSize, 1);
    dim3 grid((numOfElements + block.x - 1) / block.x, 1);

    printf("Execution configuration: <<<(%d, %d), (%d, %d)>>>\n",
           grid.x, grid.y, block.x, block.y);
    
    /* Allocate host memory */
    hostInput = (int*)calloc(numOfElements, sizeof(int));
    hostTmp = (int*)calloc(numOfElements, sizeof(int));
    hostResult = (int*)calloc(grid.x, sizeof(int));

    /* Allocate device memory */
    CHECK_CUDA_CALL(cudaMalloc((void**)&devInput, numOfBytes));
    CHECK_CUDA_CALL(cudaMalloc((void**)&devResult, grid.x * sizeof(int)));

    /* Initialize array */
    for (i = 0; i < numOfElements; ++i)
        hostInput[i] = (int)(rand() & 0xFF);

    memcpy(hostTmp, hostInput, numOfBytes);
    
    /* Execute reduction operation in host */
    gettimeofday(&startTime, NULL);
    hostSum = recursiveReduceHost(hostTmp, numOfElements);
    gettimeofday(&endTime, NULL);
    
    printf("Host execution time: %.6f, result: %d\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6),
           hostSum);
    
    /* Call reduceNeighbored kernel */
    CHECK_CUDA_CALL(cudaMemcpy(devInput, hostInput, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    gettimeofday(&startTime, NULL);
    reduceNeighbored<<<grid, block>>>(devInput, devResult, numOfElements);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());
    
    /* Copy kernel result */
    CHECK_CUDA_CALL(cudaMemcpy(hostResult, devResult,
                               grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    /* Calculate kernel result */
    devSum = 0;

    for (i = 0; i < grid.x; ++i)
        devSum += hostResult[i];

    printf("Device (reduceNeighbored) execution time: %.6f, result: %d\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6),
           devSum);
    
    if (hostSum == devSum)
        printf("Test (reduceNeighbored) passed!\n");
    else
        printf("Test (reduceNeighbored) failed!\n");

    /* Call reduceNeighboredLess kernel */
    CHECK_CUDA_CALL(cudaMemcpy(devInput, hostInput, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    gettimeofday(&startTime, NULL);
    reduceNeighboredLess<<<grid, block>>>(devInput, devResult, numOfElements);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());
    
    /* Copy kernel result */
    CHECK_CUDA_CALL(cudaMemcpy(hostResult, devResult,
                               grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    /* Calculate kernel result */
    devSum = 0;

    for (i = 0; i < grid.x; ++i)
        devSum += hostResult[i];

    printf("Device (reduceNeighboredLess) execution time: %.6f, result: %d\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6),
           devSum);
    
    if (hostSum == devSum)
        printf("Test (reduceNeighboredLess) passed!\n");
    else
        printf("Test (reduceNeighboredLess) failed!\n");
    
    /* Call reduceInterleaved kernel */
    CHECK_CUDA_CALL(cudaMemcpy(devInput, hostInput, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    gettimeofday(&startTime, NULL);
    reduceInterleaved<<<grid, block>>>(devInput, devResult, numOfElements);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());
    
    /* Copy kernel result */
    CHECK_CUDA_CALL(cudaMemcpy(hostResult, devResult,
                               grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    /* Calculate kernel result */
    devSum = 0;

    for (i = 0; i < grid.x; ++i)
        devSum += hostResult[i];

    printf("Device (reduceInterleaved) execution time: %.6f, result: %d\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6),
           devSum);
    
    if (hostSum == devSum)
        printf("Test (reduceInterleaved) passed!\n");
    else
        printf("Test (reduceInterleaved) failed!\n");
    
    /* Call reduceUnrolling2 kernel */
    CHECK_CUDA_CALL(cudaMemcpy(devInput, hostInput, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    gettimeofday(&startTime, NULL);
    reduceUnrolling2<<<grid.x / 2, block>>>(devInput, devResult, numOfElements);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());
    
    /* Copy kernel result */
    CHECK_CUDA_CALL(cudaMemcpy(hostResult, devResult,
                               grid.x / 2 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    /* Calculate kernel result */
    devSum = 0;

    for (i = 0; i < grid.x / 2; ++i)
        devSum += hostResult[i];
    
    printf("Execution configuration: <<<(%d, %d), (%d, %d)>>>\n",
           grid.x / 2, grid.y, block.x, block.y);
    printf("Device (reduceUnrolling2) execution time: %.6f, result: %d\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6),
           devSum);
    
    if (hostSum == devSum)
        printf("Test (reduceUnrolling2) passed!\n");
    else
        printf("Test (reduceUnrolling2) failed!\n");

    /* Call reduceUnrolling4 kernel */
    CHECK_CUDA_CALL(cudaMemcpy(devInput, hostInput, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    gettimeofday(&startTime, NULL);
    reduceUnrolling4<<<grid.x / 4, block>>>(devInput, devResult, numOfElements);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());
    
    /* Copy kernel result */
    CHECK_CUDA_CALL(cudaMemcpy(hostResult, devResult,
                               grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    /* Calculate kernel result */
    devSum = 0;

    for (i = 0; i < grid.x / 4; ++i)
        devSum += hostResult[i];
    
    printf("Execution configuration: <<<(%d, %d), (%d, %d)>>>\n",
           grid.x / 4, grid.y, block.x, block.y);
    printf("Device (reduceUnrolling4) execution time: %.6f, result: %d\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6),
           devSum);
    
    if (hostSum == devSum)
        printf("Test (reduceUnrolling4) passed!\n");
    else
        printf("Test (reduceUnrolling4) failed!\n");
    
    /* Call reduceUnrolling8 kernel */
    CHECK_CUDA_CALL(cudaMemcpy(devInput, hostInput, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    gettimeofday(&startTime, NULL);
    reduceUnrolling8<<<grid.x / 8, block>>>(devInput, devResult, numOfElements);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());
    
    /* Copy kernel result */
    CHECK_CUDA_CALL(cudaMemcpy(hostResult, devResult,
                               grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    /* Calculate kernel result */
    devSum = 0;

    for (i = 0; i < grid.x / 8; ++i)
        devSum += hostResult[i];
    
    printf("Execution configuration: <<<(%d, %d), (%d, %d)>>>\n",
           grid.x / 8, grid.y, block.x, block.y);
    printf("Device (reduceUnrolling8) execution time: %.6f, result: %d\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6),
           devSum);
    
    if (hostSum == devSum)
        printf("Test (reduceUnrolling8) passed!\n");
    else
        printf("Test (reduceUnrolling8) failed!\n");
    
    /* Call reduceUnrollingWarps8 kernel */
    CHECK_CUDA_CALL(cudaMemcpy(devInput, hostInput, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    gettimeofday(&startTime, NULL);
    reduceUnrollingWarps8<<<grid.x / 8, block>>>(devInput, devResult, numOfElements);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());
    
    /* Copy kernel result */
    CHECK_CUDA_CALL(cudaMemcpy(hostResult, devResult,
                               grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    /* Calculate kernel result */
    devSum = 0;

    for (i = 0; i < grid.x / 8; ++i)
        devSum += hostResult[i];
    
    printf("Execution configuration: <<<(%d, %d), (%d, %d)>>>\n",
           grid.x / 8, grid.y, block.x, block.y);
    printf("Device (reduceUnrollingWarps8) execution time: %.6f, result: %d\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6),
           devSum);
    
    if (hostSum == devSum)
        printf("Test (reduceUnrollingWarps8) passed!\n");
    else
        printf("Test (reduceUnrollingWarps8) failed!\n");
    
    /* Call reduceCompleteUnrollingWarps8 kernel */
    CHECK_CUDA_CALL(cudaMemcpy(devInput, hostInput, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    gettimeofday(&startTime, NULL);
    reduceCompleteUnrollingWarps8<<<grid.x / 8, block>>>(devInput, devResult, numOfElements);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    gettimeofday(&endTime, NULL);

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());
    
    /* Copy kernel result */
    CHECK_CUDA_CALL(cudaMemcpy(hostResult, devResult,
                               grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    /* Calculate kernel result */
    devSum = 0;

    for (i = 0; i < grid.x / 8; ++i)
        devSum += hostResult[i];
    
    printf("Execution configuration: <<<(%d, %d), (%d, %d)>>>\n",
           grid.x / 8, grid.y, block.x, block.y);
    printf("Device (reduceCompleteUnrollingWarps8) execution time: %.6f, result: %d\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6),
           devSum);
    
    if (hostSum == devSum)
        printf("Test (reduceCompleteUnrollingWarps8) passed!\n");
    else
        printf("Test (reduceCompleteUnrollingWarps8) failed!\n");
    
    /* Call reduceCompleteUnrollingWarps kernel */
    CHECK_CUDA_CALL(cudaMemcpy(devInput, hostInput, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    switch (blockSize) {
        case 1024:
            gettimeofday(&startTime, NULL);
            reduceCompleteUnrollingWarps<1024><<<grid.x / 8, block>>>(
                devInput, devResult, numOfElements);
            CHECK_CUDA_CALL(cudaDeviceSynchronize());
            gettimeofday(&endTime, NULL);
            break;

        case 512:
            gettimeofday(&startTime, NULL);
            reduceCompleteUnrollingWarps<512><<<grid.x / 8, block>>>(
                devInput, devResult, numOfElements);
            CHECK_CUDA_CALL(cudaDeviceSynchronize());
            gettimeofday(&endTime, NULL);
            break;
        
        case 256:
            gettimeofday(&startTime, NULL);
            reduceCompleteUnrollingWarps<256><<<grid.x / 8, block>>>(
                devInput, devResult, numOfElements);
            CHECK_CUDA_CALL(cudaDeviceSynchronize());
            gettimeofday(&endTime, NULL);
            break;
        
        case 128:
            gettimeofday(&startTime, NULL);
            reduceCompleteUnrollingWarps<128><<<grid.x / 8, block>>>(
                devInput, devResult, numOfElements);
            CHECK_CUDA_CALL(cudaDeviceSynchronize());
            gettimeofday(&endTime, NULL);
            break;
        
        case 64:
            gettimeofday(&startTime, NULL);
            reduceCompleteUnrollingWarps<64><<<grid.x / 8, block>>>(
                devInput, devResult, numOfElements);
            CHECK_CUDA_CALL(cudaDeviceSynchronize());
            gettimeofday(&endTime, NULL);
            break;
    }

    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());
    
    /* Copy kernel result */
    CHECK_CUDA_CALL(cudaMemcpy(hostResult, devResult,
                               grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    /* Calculate kernel result */
    devSum = 0;

    for (i = 0; i < grid.x / 8; ++i)
        devSum += hostResult[i];
    
    printf("Execution configuration: <<<(%d, %d), (%d, %d)>>>\n",
           grid.x / 8, grid.y, block.x, block.y);
    printf("Device (reduceCompleteUnrollingWarps) execution time: %.6f, result: %d\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6),
           devSum);
    
    if (hostSum == devSum)
        printf("Test (reduceCompleteUnrollingWarps) passed!\n");
    else
        printf("Test (reduceCompleteUnrollingWarps) failed!\n");

    /* Free device memory */
    CHECK_CUDA_CALL(cudaFree(devInput));
    CHECK_CUDA_CALL(cudaFree(devResult));
    
    /* Free host memory */
    free(hostInput);
    free(hostTmp);
    free(hostResult);
    
    /* Reset device */
    CHECK_CUDA_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}

