#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gtest/gtest.h"

#include "helper.h"

struct obj
{
    obj() : ptr0(nullptr), ptr1(nullptr), ptr2(nullptr)
    {
    }    
    int *ptr0, *ptr1, *ptr2;
};

__global__ void kernel(obj my_obj)
{
    printf("\n ptr0= %d, ptr1= %d, ptr2= %d",
           my_obj.ptr0[0],
           my_obj.ptr1[0],
           my_obj.ptr2[0]);
}


int main()
{
    obj my_obj;

    CUDA_ERROR(cudaMallocManaged((void**)&my_obj.ptr0, sizeof(int)));
    CUDA_ERROR(cudaMallocManaged((void**)&my_obj.ptr1, sizeof(int)));
    CUDA_ERROR(cudaMallocManaged((void**)&my_obj.ptr2, sizeof(int)));
    my_obj.ptr0[0] = 42;
    my_obj.ptr1[0] = 77;
    my_obj.ptr2[0] = 99;

    dim3 gridDim(1, 1, 1);
    dim3 blockDim(1, 1, 1);

    void* args[] = {&my_obj};
    CUDA_ERROR(cudaLaunchKernel((void*)kernel, gridDim, blockDim, args));

    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaFree(my_obj.ptr0));
    CUDA_ERROR(cudaFree(my_obj.ptr1));
    CUDA_ERROR(cudaFree(my_obj.ptr2));
}