

#ifndef relu_cuh
#define relu_cuh


__global__ void relu_forward(float * in_d, float * out_d, int n)
{
    int t= threadIdx.x + blockIdx.x * blockDim.x;
    if(t<n)out_d[t]= max(0.0, in_d[t]);
}

__global__ void relu_backward(float * dLdy_d, float * in_d , float * dLdx_d, int n)
{
    int t= threadIdx.x + blockIdx.x * blockDim.x;
    if(t<n)
    {
        if(in_d[t]>0)dLdx_d[t] = dLdy_d[t] * 1;
        else dLdx_d[t] = 0.0;

    }

}
#endif