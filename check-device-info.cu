
/* check-device-info.cu */

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
    }

int main(int argc, char** argv)
{
    int deviceCount;
    cudaError_t cudaError;

    int dev;
    cudaDeviceProp deviceProp;
    int driverVersion;
    int runtimeVersion;

    if ((cudaError = cudaGetDeviceCount(&deviceCount)) != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount() failed: %s (%d)\n",
                cudaGetErrorString(cudaError), cudaError);
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available devices(s) that support CUDA.\n");
        exit(EXIT_SUCCESS);
    }

    printf("Detected %d CUDA capable devices.\n", deviceCount);

    dev = 0;
    CHECK_CUDA_CALL(cudaSetDevice(dev));
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Device %d: %s\n", dev, deviceProp.name);

    CHECK_CUDA_CALL(cudaDriverGetVersion(&driverVersion));
    CHECK_CUDA_CALL(cudaRuntimeGetVersion(&runtimeVersion));

    printf("\tCUDA Driver version: %d.%d, Runtime version: %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    printf("\tCUDA Compute Capability: %d.%d\n",
           deviceProp.major, deviceProp.minor);

    printf("\tTotal amount of global memory: %.2f GB (%llu B)\n",
           (double)deviceProp.totalGlobalMem / pow(1024.0, 3),
           (unsigned long long)deviceProp.totalGlobalMem);

    printf("\tGPU Clock rate: %.0f MHz (%.2f GHz)\n",
           (double)deviceProp.clockRate * 1.0e-3,
           (double)deviceProp.clockRate * 1.0e-6);

    printf("\tMemory Clock rate: %.0f MHz (%.2f GHz)\n",
           (double)deviceProp.memoryClockRate * 1.0e-3,
           (double)deviceProp.memoryClockRate * 1.0e-6);

    printf("\tMemory Bus width: %d bit\n",
           deviceProp.memoryBusWidth);

    printf("\tL2 Cache size: %.2f MB (%d B)\n",
           (double)deviceProp.l2CacheSize / pow(1024.0, 2),
           deviceProp.l2CacheSize);

    printf("\tMax texture dimension size (1D): (%d)\n",
           deviceProp.maxTexture1D);

    printf("\tMax texture dimension size (2D): (%d, %d)\n",
           deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);

    printf("\tMax texture dimension size (3D): (%d, %d, %d)\n",
           deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
           deviceProp.maxTexture3D[2]);
    
    printf("\tMax layered texture size (1D): (%d) x %d\n",
           deviceProp.maxTexture1DLayered[0],
           deviceProp.maxTexture1DLayered[1]);

    printf("\tMax layered texture size (2D): (%d, %d) x %d\n",
           deviceProp.maxTexture2DLayered[0],
           deviceProp.maxTexture2DLayered[1],
           deviceProp.maxTexture2DLayered[2]);

    printf("\tTotal amount of constant memory: %.2f MB (%zu B)\n",
           (double)deviceProp.totalConstMem / pow(1024.0, 2),
           deviceProp.totalConstMem);

    printf("\tTotal amount of shared memory per block: %.2f MB (%zu B)\n",
           (double)deviceProp.sharedMemPerBlock / pow(1024.0, 2),
           deviceProp.sharedMemPerBlock);

    printf("\tTotal number of available registers per block: %d\n",
           deviceProp.regsPerBlock);

    printf("\tWarp size: %d\n", deviceProp.warpSize);

    printf("\tMaximum number of threads per multiprocessor: %d\n",
           deviceProp.maxThreadsPerMultiProcessor);

    printf("\tMaximum number of threads per block: %d\n",
           deviceProp.maxThreadsPerBlock);

    printf("\tMaximum sizes of each dimension of a block: (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);

    printf("\tMaximum sizes of each dimension of a grid: (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);

    printf("\tMaximum memory pitch: %zu B\n", deviceProp.memPitch);

    return EXIT_SUCCESS;
}

