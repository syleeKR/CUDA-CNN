
#ifndef cnn_update_cuh
#define cnn_update_cuh


__global__ void cnn_update_kernel(float * filter_d, float * dLdf_d, float lr, int n)
{
    int t= threadIdx.x + blockIdx.x * blockDim.x;
    if(t<n)filter_d[t] -= lr * dLdf_d[t];
}

#endif