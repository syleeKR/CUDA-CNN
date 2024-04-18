



#ifndef maxpool_cuh
#define maxpool_cuh


__global__ void maxpool_forward(float * x, bool * wasmax_d, float * out_d, int B, int out_dim, int Hout , int Wout)
{
    int t= threadIdx.x + blockIdx.x * blockDim.x;
    if(t<B * out_dim * Hout * Wout)
    {
        int w = t % Wout;
        int h = (t - w)/Wout%Hout;
        int c = (t - w - h*Wout)/Hout/Wout%out_dim;
        int b = t/(out_dim * Hout * Wout);

        int i1 = index_in(b,c,2*h, 2*w, B, out_dim, 2*Hout, 2*Wout); 
        int i2 = index_in(b,c,2*h+1, 2*w, B, out_dim, 2*Hout, 2*Wout); 
        int i3 = index_in(b,c,2*h, 2*w+1, B, out_dim, 2*Hout, 2*Wout); 
        int i4 = index_in(b,c,2*h+1, 2*w+1, B, out_dim, 2*Hout, 2*Wout); 

        float maxval = - INFINITY;
        int maxindex = i1;
        if (x[i1] > maxval){maxval = x[i1]; maxindex = i1;}
        if (x[i2] > maxval){maxval = x[i2]; maxindex = i2;}
        if (x[i3] > maxval){maxval = x[i3]; maxindex = i3;}
        if (x[i4] > maxval){maxval = x[i4]; maxindex = i4;}

        wasmax_d[maxindex] = true;
        out_d[t] = maxval;
    }
}

__global__ void maxpool_backward(float * dLdy_d, bool * wasmax_d , float * dLdx_d, int B, int C, int H, int W)
{
    int t= threadIdx.x + blockIdx.x * blockDim.x;
    if(t<B*C*H*W)
    {
        if (wasmax_d[t]==false)dLdx_d[t] = 0.0;
        else
        {
            int w = t % W;
            int h = (t - w)/W%H;
            int c = (t - w - h*W)/H/W%C;
            int b = t/(C * H * W);

            dLdx_d[t] = dLdy_d[index_in(b,c,h/2,w/2, B,C,H/2, W/2)];

        }

    }

}
#endif