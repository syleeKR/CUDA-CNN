
#ifndef cnn_cuh
#define cnn_cuh

#define TILE_SIZE 16

__device__ int index_in(int i, int j, int k, int l, int B, int in_dim, int H, int W)
{
    return in_dim*H*W*i + H*W *j + W *k + l;
}
__device__ int index_out(int i, int j, int k, int l, int B, int out_dim, int Hout, int Wout)
{

    return out_dim*Hout*Wout*i + Hout*Wout *j + Wout *k + l;
}
__device__ int index_filter(int i, int j, int k, int l, int out_dim , int in_dim, int filter_size)
{
    // Cout, Cin, K, K
    return i *(in_dim * filter_size * filter_size) + j *(filter_size * filter_size) + k*filter_size + l;
}
__global__ void forward_kernel(float * in_d, float * filter_d, int B, int out_dim, int in_dim, int filter_size, int H, int W, int Hout, int Wout, float * out_d)
{
    int c_o = blockIdx.x;
    int b = blockIdx.z;
    int blockperrow = ceil((float)Wout/TILE_SIZE);
    int h = blockIdx.y / blockperrow * TILE_SIZE + threadIdx.y;
    int w = (blockIdx.y % blockperrow) * TILE_SIZE + threadIdx.x;
    float sum = 0.0;
    if (0 <= h && h< Hout && 0<=w && w < Wout)
    {
        REP0(c_i, in_dim)
        {
            REP0(k1, filter_size)
            {
                REP0(k2, filter_size)
                {
                    sum += filter_d[index_filter(c_o,c_i,k1,k2,out_dim, in_dim, filter_size)] * in_d[index_in(b,c_i, h+k1, w+k2, B, in_dim, H, W)];
                }
            }
        }
        out_d[index_out(b, c_o, h,w, B, out_dim , Hout, Wout)] = sum;
    }
}

__global__ void backward_kernel_dLdf(float * in_d, float * dLdy_d, int B, int out_dim, int in_dim, int filter_size, int H, int W, int Hout, int Wout, float * dLdf_d)
{
    int c_o = blockIdx.x/in_dim;
    int c_i = blockIdx.x%in_dim;
    int i = blockIdx.y / filter_size;
    int j = blockIdx.y % filter_size;

    float sum = 0.0;

    REP0(b,B){
        for(int k = threadIdx.y ; k<Hout; k+= blockDim.y)
        {
            for(int l =threadIdx.x; l<Wout; l+= blockDim.x)
            {
                sum += dLdy_d[index_out(b,c_o, k, l, B, out_dim, Hout, Wout)] * in_d[index_in(b,c_i, k+i, j+l, B, in_dim, H,W)];
            }
        }
    }
    atomicAdd(&dLdf_d[index_filter(c_o, c_i, i, j, out_dim, in_dim, filter_size)], sum); 
}

__global__ void backward_kernel_dLdx(float * filter_d, float * dLdy_d, int B, int out_dim, int in_dim, int filter_size,int H, int W, int Hout, int Wout, float * dLdx_d)
{
    int c_i = blockIdx.x;
    int b = blockIdx.z;
    int blockperrow = ceil((float)W/TILE_SIZE);
    int h = blockIdx.y / blockperrow * TILE_SIZE + threadIdx.y;
    int w = (blockIdx.y % blockperrow) * TILE_SIZE + threadIdx.x;
    float sum = 0.0;

    REP0(i, out_dim)
    {
        REP0(j, filter_size)
        {
            REP0(k, filter_size)
            {
                if (h>=j && w>=k && h-j <Hout && w -k <Wout)
                {
                    sum += filter_d[index_filter(i, c_i, j, k, out_dim, in_dim, filter_size)] * dLdy_d[index_out(b, i, h-j, w-k, B, out_dim, Hout, Wout)];
                }
            }
        }
    }
    dLdx_d[index_in(b,c_i,h,w, B, in_dim , H, W)] = sum;
}

#endif