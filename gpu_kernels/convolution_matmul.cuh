#ifndef cnn_matmul_cuh
#define cnn_matmul_cuh


__global__ void mm_unroll(float * in_d, int B, int out_dim, int in_dim, int filter_size, int H, int W, int Hout, int Wout, float * unrolled_d)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int n_col = B * Hout * Wout ;
    if (t < in_dim * n_col)
    {
        int c = t / n_col;
        int col_location = t % n_col;

        int w = col_location%Wout;
        int h = (col_location-w)/Wout%Hout;
        int b = (col_location-w-h*Wout)/Hout/Wout;

        int row_start = c * filter_size *filter_size;

        REP0(k1, filter_size)
        {
            REP0(k2, filter_size)
            {
                unrolled_d[(B*Hout*Wout)*(row_start + k1 * filter_size + k2) + (col_location)] = in_d[index_in(b,c,h+k1, w+k2, B,in_dim, H,W)];
            }
        }
    }
}


__global__ void mm_transpose(float * temp_d, int B, int out_dim, int in_dim, int filter_size, int H, int W, int Hout, int Wout, float *out_d)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int n_col = B * Hout * Wout;
    if(t < out_dim * n_col)
    {
        int c = t / n_col;
        int col_location = t % n_col;
        int w = col_location%Wout;
        int h = (col_location-w)/Wout%Hout;
        int b = (col_location-w-h*Wout)/Hout/Wout;
        
        int new_index = index_out(b,c,h,w, B, out_dim, Hout, Wout);
        out_d[new_index] = temp_d[t];
    }
}



__global__
void mm_matmul(float * a , float *b, int n, int m, int k, float *c)
{
    // (n.m) times (m.k) matmul
  __shared__ float atile[TILE_SIZE][TILE_SIZE];
  __shared__ float btile[TILE_SIZE][TILE_SIZE];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty= threadIdx.y;
  int col = bx*blockDim.x + tx;
  int row = by*blockDim.y + ty;

  float val = 0.0;
  for(int rep=0; rep<ceil((float)m/TILE_SIZE); rep++)
  {
  
    if(row<n && (rep*TILE_SIZE + tx) < m)
    {
      atile[ty][tx] = a[row * m +  rep*TILE_SIZE + tx];
    }
    else atile[ty][tx] = 0.0f;
    if (rep*TILE_SIZE + ty < m && col <k)
    {
      btile[ty][tx] = b[k*(rep*TILE_SIZE + ty) + col];
    }
    else btile[ty][tx] = 0.0f;
    
    __syncthreads();

    for(int i=0 ; i<TILE_SIZE ; i++)
    {
      val += atile[ty][i] * btile[i][tx];
    }
    __syncthreads();

  }
  if ((row < n) && (col <k))c[row * k + col] = val;
}


#endif
