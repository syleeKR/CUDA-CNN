

#ifndef maxpool_hpp
#define maxpool_hpp


class MAXPOOL
{
    public:
        string mode;
        int B,C,H,W;
        float * in = nullptr;
        float * out = nullptr;
        float * dLdx = nullptr;
        bool * wasmax = nullptr;

        float * in_d = nullptr;
        float * out_d = nullptr;
        float * dLdx_d = nullptr;
        bool * wasmax_d = nullptr;

        MAXPOOL(){}
        MAXPOOL(vint & input_size, string mode):mode(mode)
        {
            B = input_size[0];
            C = input_size[1];
            H = input_size[2];
            W = input_size[3];

            if (mode == "gpu_optimized"  || mode =="gpu_naive")
            {
                cudaMalloc((void **)&in_d, sizeof(float) * B*C*H*W);
                cudaMalloc((void **)&out_d, sizeof(float) * B*C*(H/2)*(W/2));
                cudaMalloc((void **)&dLdx_d, sizeof(float) * B*C*H*W);
                cudaMalloc((void **)&wasmax_d, sizeof(bool) * B*C*H*W);

            }
            else{
                in = new float[B*C*H*W];
                wasmax = new bool[B*C*H*W];
                out = new float[B*C*(H/2)*(W/2)];
                dLdx = new float[B*C*H*W];
            }
        }
        ~MAXPOOL()
        {
            if (mode == "gpu_optimized" || mode == "gpu_naive")
            {
                cudaFree(in_d);
                cudaFree(out_d);
                cudaFree(dLdx_d);
                cudaFree(wasmax_d);
            }
            else
            {
                delete [] in;
                delete [] out;
                delete [] dLdx;
                delete [] wasmax;
            }
        }
        int index(int i, int j, int k, int l)
        {
            return C*H*W*i + H*W *j + W *k + l;
        }
        int index_out(int i, int j, int k, int l)
        {
            return C*H/2*W/2*i + H/2*W/2 *j + W/2 *k + l;
        }
        void forward(float * x)
        {
            if (mode == "gpu_optimized" || mode == "gpu_naive")forward_gpu(x);
            else forward_cpu(x);
        }
        void backward(float * dLdy)
        {
            if (mode == "gpu_optimized" || mode == "gpu_naive")backward_gpu(dLdy);
            else backward_cpu(dLdy);
        }

        void forward_gpu(float * x)
        {
            cudaMemcpy(in_d, x, B*C*H*W*sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemset(wasmax_d, 0, sizeof(bool) * B * C*H*W);
            dim3 a(ceil((float)B*C*(H/2) * (W/2)/1024),1,1);
            dim3 b(1024,1,1);
            maxpool_forward<<<a,b>>>(in_d, wasmax_d, out_d, B, C, H/2,W/2 );
        }
        void backward_gpu(float * dLdy_d)
        {
            dim3 a(ceil((float)B*C*H*W/1024),1,1);
            dim3 b(1024,1,1);
            maxpool_backward<<<a,b>>>(dLdy_d, wasmax_d, dLdx_d, B, C, H,W );
        }


        void forward_cpu(float * x)
        {
            REP0(i, B)
            {
                REP0(j, C)
                {
                    REP0(k, H/2)
                    {
                        REP0(l, W/2)
                        {
                            int p1 = index(i,j,2*k,2*l);
                            int p2 = index(i,j,2*k+1,2*l);
                            int p3 = index(i,j,2*k,2*l+1);
                            int p4 = index(i,j,2*k+1,2*l+1);
                            vint tmp = {p1,p2,p3,p4};

                            float maxval = - INFINITY;
                            float maxindex = p1;
                            REP0(curr, sz(tmp))
                            {
                                in[tmp[curr]] = x[tmp[curr]];
                                if(x[tmp[curr]] > maxval)
                                {
                                    maxval = x[tmp[curr]];
                                    maxindex = tmp[curr];
                                }
                            }
                            REP0(curr, sz(tmp))
                            {
                                if (maxindex == tmp[curr])
                                {
                                    wasmax[tmp[curr]] = true;
                                }
                                else
                                {
                                    wasmax[tmp[curr]] = false;
                                }
                            }
                            out[index_out(i,j,k,l)] = maxval;
                        }
                    }
                }
            }

        }

        void backward_cpu(float * dLdy)
        {
            REP0(i, B)
            {
                REP0(j, C)
                {
                    REP0(k, H)
                    {
                        REP0(l, W)
                        {
                            int p = index(i,j,k,l);
                            if (wasmax[p]==false)dLdx[p]=0.0;
                            else
                            {
                                dLdx[p] = dLdy[index_out(i,j,k/2, l/2)];
                            }
                        }
                    }
                }
            }
        }
};




#endif
