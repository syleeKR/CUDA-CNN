


#ifndef cnn_hpp
#define cnn_hpp


class CNN
{
    public:
        string mode;
        int B,C,H,W;
        int Hout, Wout;
        int in_dim, out_dim, filter_size;


        float * in = nullptr;
        float * out = nullptr;
        float * dLdx = nullptr;
        float * filter =nullptr;
        float * dLdf = nullptr;

        float * in_d = nullptr;
        float * out_d = nullptr;
        float * dLdx_d = nullptr;
        float * filter_d = nullptr;
        float * dLdf_d = nullptr;

        int index_in(int i, int j, int k, int l)
        {
            return in_dim*H*W*i + H*W *j + W *k + l;
        }
        int index_out(int i, int j, int k, int l)
        {

            return out_dim*Hout*Wout*i + Hout*Wout *j + Wout *k + l;
        }
        int index_filter(int i, int j, int k, int l)
        {
            // Cout, Cin, K, K
            return i *(in_dim * filter_size * filter_size) + j *(filter_size * filter_size) + k*filter_size + l;
        }
        CNN(){}
        CNN(int in_dim, int out_dim, int filter_size, vint & input_size, string mode):in_dim(in_dim), out_dim(out_dim), filter_size(filter_size), mode(mode)
        {
            B = input_size[0];
            C = input_size[1];
            H = input_size[2];
            W = input_size[3];
            Hout = H - filter_size + 1;
            Wout = W - filter_size + 1;

            if (mode == "gpu_optimized" || mode == "gpu_naive")
            {
                cudaMalloc((void **)&in_d, sizeof(float) * B*in_dim*H*W);
                cudaMalloc((void **)&out_d, sizeof(float) * B*out_dim*Hout *Wout);
                cudaMalloc((void **)&dLdx_d, sizeof(float) * B*C*H*W);
                cudaMalloc((void **)&filter_d, sizeof(float) * out_dim * in_dim * filter_size * filter_size);
                cudaMalloc((void **)&dLdf_d, sizeof(float) * out_dim * in_dim * filter_size * filter_size);
                filter = new float[out_dim * in_dim * filter_size * filter_size];

            }
            else
            {
                in = new float[B*in_dim*H*W];
                out = new float[B*out_dim*Hout *Wout];
                dLdx = new float[B*C*H*W];
                filter = new float[out_dim * in_dim * filter_size * filter_size];
                dLdf = new float[out_dim * in_dim * filter_size * filter_size];

            }

            std::random_device rd;  
            std::mt19937 gen(rd()); 
            int ntot =(out_dim * in_dim * filter_size * filter_size);
            float bound = sqrt(6.0/(float)ntot);
            std::uniform_real_distribution<float> dis(-bound, +bound); 
            REP0(i, out_dim * in_dim * filter_size * filter_size){filter[i] =dis(gen);}

            if (mode == "gpu_optimized" || mode == "gpu_naive")
            {
                cudaMemcpy(filter_d, filter, sizeof(float)*out_dim *in_dim *filter_size *filter_size, cudaMemcpyHostToDevice);
                delete [] filter;
            }
        }
        ~CNN()
        {
            if (mode == "gpu_optimized" || mode == "gpu_naive")
            {
                cudaFree(in_d);
                cudaFree(out_d);
                cudaFree(dLdx_d);
                cudaFree(filter_d);
                cudaFree(dLdf_d);
            }
            else
            {
                delete [] in;
                delete [] out;
                delete [] dLdx;
                delete [] filter;
                delete [] dLdf;
            }
        }
        void forward(float * x)
        {
            if (mode == "gpu_naive")forward_gpu(x);
            else if(mode == "gpu_optimized")forward_gpu_mm(x);
            else forward_cpu(x);
        }
        void backward(float * dLdy)
        {
            if (mode == "gpu_naive")backward_gpu(dLdy);
            else if(mode == "gpu_optimized")backward_gpu_mm(dLdy);
            else backward_cpu(dLdy);
        }
        void forward_gpu_mm(float * x, bool use_cublas= false)
        {
            // copy to device
            //memcpy(in, x, B*in_dim*H*W*sizeof(float));
            //cudaMemcpy(filter_d, filter , sizeof(float) * out_dim * in_dim * filter_size * filter_size, cudaMemcpyHostToDevice);
            //cudaMemcpy(in_d, in, sizeof(float)*B*in_dim*H*W, cudaMemcpyHostToDevice);
            cudaMemcpy(in_d, x, B*C*H*W*sizeof(float), cudaMemcpyDeviceToDevice);
            // Step 1 : unroll please
            float * unrolled_d;
            cudaMalloc((void **)&unrolled_d, sizeof(float)*B*in_dim*filter_size*filter_size*Hout*Wout);
            dim3 dimofgrid_unroll(ceil((float)B * Hout*Wout*in_dim / 1024),1,1);
            dim3 dimofblock_unroll(1024,1,1);
            mm_unroll<<<dimofgrid_unroll, dimofblock_unroll>>>(in_d, B, out_dim, in_dim, filter_size, H, W, Hout, Wout, unrolled_d);
            
            // Step 2 : Matmul (cout Cin*f*f) . (Cin*f*f, B*Hout*Wout)
            float * temp_d; 
            cudaMalloc((void **)&temp_d, sizeof(float)*B*out_dim*Hout*Wout);

            if (use_cublas)
            {
                cublas_matmul(filter_d, unrolled_d,temp_d,out_dim, in_dim * filter_size*filter_size, B * Hout*Wout);
            }
            else{
                dim3 dimofgrid_matmtul( ceil((float)(B*Hout*Wout)/TILE_SIZE), ceil((float)out_dim/TILE_SIZE));
                dim3 dimofblock_matmul(TILE_SIZE,TILE_SIZE);
                mm_matmul<<<dimofgrid_matmtul, dimofblock_matmul>>>(filter_d, unrolled_d, out_dim, in_dim * filter_size*filter_size, B * Hout*Wout, temp_d);
            }
            // Step 3 : Transpose (Cout B*Hout*Wout) -> (B Cout Hout Wout) 
            dim3 dimofgrid_transpose(ceil((float)B * out_dim * Hout*Wout / 1024),1,1);
            dim3 dimofblock_transpose(1024,1,1);
            mm_transpose<<<dimofgrid_transpose, dimofblock_transpose>>>(temp_d, B, out_dim, in_dim, filter_size, H, W, Hout, Wout, out_d);
            
            // Copy and free memory
            //cudaMemcpy(out, out_d, sizeof(float) * B * out_dim * Hout * Wout, cudaMemcpyDeviceToHost);
            cudaFree(temp_d);
            cudaFree(unrolled_d);
        }

        void forward_gpu(float * x)
        {
            cudaMemcpy(in_d, x, B*C*H*W*sizeof(float), cudaMemcpyDeviceToDevice);
            // in
            //memcpy(in, x, B*in_dim*H*W*sizeof(float));
            // out : convolution
            //cudaMemcpy(in_d, in, sizeof(float)*B*in_dim*H*W, cudaMemcpyHostToDevice);
            //cudaMemcpy(filter_d, filter , sizeof(float) * out_dim * in_dim * filter_size * filter_size, cudaMemcpyHostToDevice);
            dim3 dimofgrid(out_dim, (int)ceil((float)Hout/TILE_SIZE) * (int)ceil((float)Wout/TILE_SIZE)  ,B);
            dim3 dimofblock(TILE_SIZE,TILE_SIZE,1);
            forward_kernel<<<dimofgrid, dimofblock>>>(in_d, filter_d, B, out_dim, in_dim, filter_size, H, W, Hout, Wout, out_d);
            //cudaMemcpy(out, out_d, sizeof(float) * B * out_dim * Hout * Wout, cudaMemcpyDeviceToHost);
        }


        void backward_gpu_mm(float * dLdy_d, bool use_cublas = false)
        {
            // cudaMemcpy all the necessary stuff
            //float * dLdy_d;
            //cudaMalloc((void **)&dLdy_d, sizeof(float)*B*out_dim*Hout*Wout);
            //cudaMemcpy(in_d, in, sizeof(float)*B*in_dim*H*W, cudaMemcpyHostToDevice);
            //cudaMemcpy(dLdy_d, dLdy , sizeof(float)*B*out_dim*Hout*Wout, cudaMemcpyHostToDevice);
            //cudaMemcpy(filter_d, filter , sizeof(float)*out_dim*in_dim *filter_size*filter_size, cudaMemcpyHostToDevice);

            
            //calc dLdf
            cudaMemset(dLdf_d, 0.0, sizeof(float) * out_dim* in_dim *filter_size*filter_size);
            dim3 dimofgrid_dLdf(out_dim*in_dim, filter_size * filter_size);
            dim3 dimofblock_dLdf(TILE_SIZE,TILE_SIZE,1);
            backward_kernel_dLdf<<<dimofgrid_dLdf, dimofblock_dLdf>>>(in_d, dLdy_d, B, out_dim, in_dim, filter_size, H, W, Hout, Wout, dLdf_d);
            //cudaMemcpy(dLdf, dLdf_d,  sizeof(float) * out_dim * in_dim * filter_size * filter_size, cudaMemcpyDeviceToHost);
            
            
            //calc dLdx
            // Step 1 : unroll please
            float * unrolled_d;
            cudaMalloc((void **)&unrolled_d, sizeof(float)*out_dim*filter_size*filter_size*B*H*W);
            dim3 dimofgrid_unroll_back(ceil((float)B * H*W*out_dim / 1024),1,1);
            dim3 dimofblock_unroll_back(1024,1,1);
            mm_unroll_back<<<dimofgrid_unroll_back, dimofblock_unroll_back>>>(dLdy_d, B, out_dim, in_dim, filter_size, H, W, Hout, Wout, unrolled_d);

            // Step 2: Transpose filter
            float * filter_transposed_d; 
            cudaMalloc((void **)&filter_transposed_d, sizeof(float)*out_dim*in_dim*filter_size * filter_size);

            dim3 dimofgrid_transpose_filter_back(ceil((float)out_dim* in_dim * filter_size * filter_size / 1024),1,1);
            dim3 dimofblock_transpose_filter_back(1024,1,1);
            mm_transpose_filter_back<<<dimofgrid_transpose_filter_back, dimofblock_transpose_filter_back>>>(filter_d, B, out_dim, in_dim, filter_size, H, W, Hout, Wout, filter_transposed_d);

            // Step 3 : Matmul (cin Cout*f*f) . (Cout*f*f, B*H*W)
            float * temp_d; 
            cudaMalloc((void **)&temp_d, sizeof(float)*B*in_dim*H*W);

            if (use_cublas)
            {
                cublas_matmul(filter_transposed_d, unrolled_d,temp_d,in_dim, out_dim * filter_size*filter_size, B * H*W);
            }
            else{
                dim3 dimofgrid_matmtul_back( ceil((float)(B*H*W)/TILE_SIZE), ceil((float)in_dim/TILE_SIZE));
                dim3 dimofblock_matmul_back(TILE_SIZE,TILE_SIZE);
                mm_matmul<<<dimofgrid_matmtul_back, dimofblock_matmul_back>>>(filter_transposed_d, unrolled_d, in_dim, out_dim * filter_size*filter_size, B * H*W, temp_d);
            }
            // Step 3 : Transpose (Cin B*H*W) -> (B Cin H W) 
            dim3 dimofgrid_transpose_back(ceil((float)B * in_dim * H*W / 1024),1,1);
            dim3 dimofblock_transpose_back(1024,1,1);
            mm_transpose_in_back<<<dimofgrid_transpose_back, dimofblock_transpose_back>>>(temp_d, B, out_dim, in_dim, filter_size, H, W, Hout, Wout, dLdx_d);

            // Copy and free memory
            //cudaMemcpy(dLdx, dLdx_d, sizeof(float) * B * in_dim * H * W, cudaMemcpyDeviceToHost);
            cudaFree(temp_d);
            cudaFree(unrolled_d);
            cudaFree(filter_transposed_d);

        }
        void backward_gpu(float * dLdy_d)
        {
            // cudaMemcpy all the necessary stuff
            //float * dLdy_d;
            //cudaMalloc((void **)&dLdy_d, sizeof(float)*B*out_dim*Hout*Wout);

            //cudaMemcpy(in_d, in, sizeof(float)*B*in_dim*H*W, cudaMemcpyHostToDevice);
            //cudaMemcpy(dLdy_d, dLdy , sizeof(float)*B*out_dim*Hout*Wout, cudaMemcpyHostToDevice);
            //cudaMemcpy(filter_d, filter , sizeof(float)*out_dim*in_dim *filter_size*filter_size, cudaMemcpyHostToDevice);

            
            //calc dLdf
            cudaMemset(dLdf_d, 0.0, sizeof(float) * out_dim* in_dim *filter_size*filter_size);
            dim3 dimofgrid_dLdf(out_dim*in_dim, filter_size * filter_size);
            dim3 dimofblock_dLdf(TILE_SIZE,TILE_SIZE,1);
            backward_kernel_dLdf<<<dimofgrid_dLdf, dimofblock_dLdf>>>(in_d, dLdy_d, B, out_dim, in_dim, filter_size, H, W, Hout, Wout, dLdf_d);
            //cudaMemcpy(dLdf, dLdf_d,  sizeof(float) * out_dim * in_dim * filter_size * filter_size, cudaMemcpyDeviceToHost);
            
            
            //calc dLdx
            dim3 dimofgrid_dLdx(in_dim, (int)ceil((float)H/TILE_SIZE) * (int)ceil((float)W/TILE_SIZE)  ,B);
            dim3 dimofblock_dLdx(TILE_SIZE,TILE_SIZE,1);
            backward_kernel_dLdx<<<dimofgrid_dLdx, dimofblock_dLdx>>>(filter_d, dLdy_d, B, out_dim, in_dim, filter_size, H, W, Hout, Wout, dLdx_d);
            //cudaMemcpy(dLdx, dLdx_d,  sizeof(float) * B * in_dim * H * W, cudaMemcpyToHost);
            
        }

        
        void forward_cpu(float * x)
        {
            REP0(i,B)
            {
                REP0(j, out_dim)
                {
                    REP0(k, Hout)
                    {
                        REP0(l, Wout)
                        {
                            int outindex = index_out(i,j,k,l);
                            // calculate out[outindex] here
                            out[outindex] = 0.0;
                            REP0(c, in_dim)
                            {
                                REP0(k1, filter_size)
                                {
                                    REP0(k2, filter_size)
                                    {
                                        out[outindex] += filter[index_filter(j,c,k1,k2)] * x[index_in(i,c, k+k1, l+k2)];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //store in for back prop
            memcpy(in, x, B*in_dim*H*W*sizeof(float));
        }

        void backward_cpu(float * dLdy)
        {
            // calculate dLdf

            REP0(c_o, out_dim)
            {
                REP0(c_i, in_dim)
                {
                    REP0(i, filter_size)
                    {
                        REP0(j, filter_size)
                        {
                            int index = index_filter(c_o, c_i, i, j);
                            dLdf[index] = 0.0;

                            REP0(b, B)
                            {
                                REP0(k, Hout)
                                {
                                    REP0(l, Wout)
                                    {
                                        dLdf[index] += dLdy[index_out(b,c_o, k,l)] * in[index_in(b, c_i, k+i, j+l)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            //calculate dLdx
            REP0(b, B)
            {
                REP0(c, C)
                {
                    REP0(h, H)
                    {
                        REP0(w, W)
                        {
                            int index = index_in(b,c,h,w);
                            dLdx[index]  = 0.0;
                            REP0(i, out_dim)
                            {
                                REP0(j, filter_size)
                                {
                                    REP0(k, filter_size)
                                    {
                                        if (h>=j && w>=k && h-j <Hout && w -k <Wout)
                                        {
                                            dLdx[index] += filter[index_filter(i, c, j, k)] * dLdy[index_out(b, i, h-j, w-k)];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        void update(float lr)
        {
            if (mode == "cpu"){
                REP0(i, out_dim *in_dim*filter_size * filter_size){
                    filter[i] -= lr * dLdf[i];
                }
            }
            else
            {
                int filter_n = out_dim * in_dim *filter_size * filter_size;
                dim3 a(ceil((float)filter_n/1024),1,1);
                dim3 b(1024,1,1);
                cnn_update_kernel<<<a,b>>>(filter_d, dLdf_d, lr, filter_n);
            }
        }


};

#endif

