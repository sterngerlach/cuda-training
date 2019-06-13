
/* check-dimension.cu */

#include <stdio.h>
#include <stdlib.h>

__global__ void checkDimension(void)
{
    printf("threadIdx: (%d, %d, %d), blockIdx: (%d, %d, %d), "
           "blockDim: (%d, %d, %d), gridDim: (%d, %d, %d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char** argv)
{
    int numOfElements = 6;

    dim3 block(3);
    dim3 grid((numOfElements + block.x - 1) / block.x);
    
    printf("grid.x: %d, grid.y: %d, grid.z: %d\n",
           grid.x, grid.y, grid.z);
    printf("block.x: %d, block.y: %d, block.z: %d\n",
           block.x, block.y, block.z);

    checkDimension<<<grid, block>>>();

    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}

