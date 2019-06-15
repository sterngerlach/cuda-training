
/* hello-world.cu */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void helloWorldFromGPU(void)
{
    printf("Hello, World from GPU thread %d!\n", threadIdx.x);
}

int main(int argc, char** argv)
{
    printf("Hello, World from CPU!\n");

    helloWorldFromGPU<<<1, 10>>>();

    /* cudaDeviceReset(); */
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}

