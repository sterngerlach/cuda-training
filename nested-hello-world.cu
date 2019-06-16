
/* nested-hello-world.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/*
 * Compile the program using the following command
 *
 * nvcc -arch=sm_35 -rdc=true -lcudadevrt -o bin/nested-hello-world nested-hello-world.cu
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

__global__ void nestedHelloWorld(int numOfThreads, int depthLevel)
{
    printf("Recursion level: %d, Hello World from thread %d, block %d\n",
           depthLevel, threadIdx.x, blockIdx.x);

    if (numOfThreads == 1)
        return;
    
    numOfThreads >>= 1;

    if (threadIdx.x == 0 && blockIdx.x == 0 && numOfThreads > 0) {
        nestedHelloWorld<<<2, numOfThreads>>>(numOfThreads, ++depthLevel);
        printf("Nested execution depth: %d\n", depthLevel);
    }
}

int main(int argc, char** argv)
{
    /* Set execution configuration */
    dim3 block(8, 1);
    dim3 grid(2, 1);
    
    printf("Execution configuration: <<<(%d, %d), (%d, %d)>>>\n",
           grid.x, grid.y, block.x, block.y);

    nestedHelloWorld<<<grid, block>>>(block.x, 0);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    
    /* Check kernel error */
    CHECK_CUDA_CALL(cudaGetLastError());
    
    /* Reset device */
    CHECK_CUDA_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}

