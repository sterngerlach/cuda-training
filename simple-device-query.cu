
/* simple-device-query.cu */

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

int main(int argc, char** argv)
{
    int dev;
    cudaDeviceProp deviceProp;

    /* Get device properties */
    dev = 0;
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Using device %d: %s\n", dev, deviceProp.name);
    
    printf("\tNumber of multiprocessors: %d\n",
           deviceProp.multiProcessorCount);

    printf("\tTotal amount of constant memory: %5.2f KB\n",
           (double)deviceProp.totalConstMem / 1024.0);

    printf("\tTotal amount of shared memory per block: %5.2f KB\n",
           (double)deviceProp.sharedMemPerBlock / 1024.0);

    printf("\tTotal number of available registers per block: %d\n",
           deviceProp.regsPerBlock);

    printf("\tWarp size: %d\n", deviceProp.warpSize);

    printf("\tMaximum number of threads per block: %d\n",
           deviceProp.maxThreadsPerBlock);

    printf("\tMaximum number of threads per multiprocessor: %d\n",
           deviceProp.maxThreadsPerMultiProcessor);

    printf("\tMaximum number of warps per multiprocessor: %d\n",
           deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize);

    return EXIT_SUCCESS;
}

