
/* nested-reduce.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/*
 * Compile the program using the following command
 *
 * nvcc -arch=sm_35 -rdc=true -lcudadevrt -o bin/nested-reduce nested-reduce.cu
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

__global__ void recursiveReduceGPU(int* inArray, int* outArray, unsigned int blockSize)
{
    /* Set thread id */
    unsigned int id = threadIdx.x;

    /* Convert global data pointer to local pointer for this block */
    int* pInArray = inArray + blockIdx.x * blockDim.x;
    int* pOutArray = &outArray[blockIdx.x];

    int stride;

    if (blockSize == 2 && id == 0) {
        outArray[blockIdx.x] = pInArray[0] + pInArray[1];
        return;
    }
    
    /* Nested kernel call */
    stride = blockSize >> 1;
    
    /* Perform in-place reduction */
    if (stride > 1 && id < stride)
        pInArray[id] += pInArray[id + stride];
    
    /* Synchronize between all threads in the thread block */
    /* Actually, these synchronizations are unnecessary for this kernel */
    __syncthreads();
    
    /* Nested kernel call to create child grids */
    if (id == 0) {
        recursiveReduceGPU<<<1, stride>>>(pInArray, pOutArray, stride);

        /* Synchronize between all child grids called by this block */
        cudaDeviceSynchronize();
    }
    
    /* Synchronize between all threads in the thread block */
    __syncthreads();
}

__global__ void recursiveReduceNoSync(int* inArray, int* outArray, unsigned int blockSize)
{
    /* Set thread id */
    unsigned int id = threadIdx.x;

    /* Convert global data pointer to local pointer for this block */
    int* pInArray = inArray + blockIdx.x * blockDim.x;
    int* pOutArray = &outArray[blockIdx.x];

    int stride;

    if (blockSize == 2 && id == 0) {
        outArray[blockIdx.x] = pInArray[0] + pInArray[1];
        return;
    }
    
    /* Nested kernel call */
    stride = blockSize >> 1;
    
    /* Perform in-place reduction */
    if (stride > 1 && id < stride)
        pInArray[id] += pInArray[id + stride];
    
    /* Nested kernel call to create child grids */
    if (id == 0)
        recursiveReduceGPU<<<1, stride>>>(pInArray, pOutArray, stride);
}

__global__ void recursiveReduceGPU2(int* inArray, int* outArray, int stride, int dim)
{
    /* Convert global data pointer to local pointer for this block */
    int* pInArray = inArray + blockIdx.x * dim;

    if (stride == 1 && threadIdx.x == 0) {
        outArray[blockIdx.x] = pInArray[0] + pInArray[1];
        return;
    }

    /* Perform in-place reduction */
    pInArray[threadIdx.x] += pInArray[threadIdx.x + stride];

    /* Nested kernel call to create child grids */
    if (threadIdx.x == 0 && blockIdx.x == 0)
        recursiveReduceGPU2<<<gridDim.x, stride / 2>>>(
            inArray, outArray, stride / 2, dim);
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
    numOfElements = 1 << 20;
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
   
    /* Call recursiveReduceGPU kernel */
    CHECK_CUDA_CALL(cudaMemcpy(devInput, hostInput, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    gettimeofday(&startTime, NULL);
    recursiveReduceGPU<<<grid, block>>>(devInput, devResult, blockSize);
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

    printf("Device (recursiveReduceGPU) execution time: %.6f, result: %d\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6),
           devSum);
    
    if (hostSum == devSum)
        printf("Test (recursiveReduceGPU) passed!\n");
    else
        printf("Test (recursiveReduceGPU) failed!\n");
    
    /* Call recursiveReduceNoSync kernel */
    CHECK_CUDA_CALL(cudaMemcpy(devInput, hostInput, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    gettimeofday(&startTime, NULL);
    recursiveReduceNoSync<<<grid, block>>>(devInput, devResult, blockSize);
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

    printf("Device (recursiveReduceNoSync) execution time: %.6f, result: %d\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6),
           devSum);
    
    if (hostSum == devSum)
        printf("Test (recursiveReduceNoSync) passed!\n");
    else
        printf("Test (recursiveReduceNoSync) failed!\n");
    
    /* Call recursiveReduceGPU2 kernel */
    CHECK_CUDA_CALL(cudaMemcpy(devInput, hostInput, numOfBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    gettimeofday(&startTime, NULL);
    recursiveReduceGPU2<<<grid, block>>>(devInput, devResult, blockSize / 2, block.x);
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

    printf("Device (recursiveReduceGPU2) execution time: %.6f, result: %d\n",
           ((double)endTime.tv_sec + (double)endTime.tv_usec * 1.0e-6) -
           ((double)startTime.tv_sec + (double)startTime.tv_usec * 1.0e-6),
           devSum);
    
    if (hostSum == devSum)
        printf("Test (recursiveReduceGPU2) passed!\n");
    else
        printf("Test (recursiveReduceGPU2) failed!\n");

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

