

#ifndef relu_hpp
#define relu_hpp

class RELU
{
    public:
        int n;
        string mode;
        float * in = nullptr;
        float * out = nullptr;
        float * dLdx = nullptr;

        float * in_d = nullptr;
        float * out_d = nullptr;
        float * dLdx_d = nullptr;

        RELU(){}
        RELU(int n, string mode):n(n), mode(mode)
        {
            if (mode == "gpu_optimized" || mode=="gpu_naive")
            {
                cudaMalloc((void **)&in_d, sizeof(float) * n);
                cudaMalloc((void **)&out_d, sizeof(float) * n);
                cudaMalloc((void **)&dLdx_d, sizeof(float) * n);
            }
            else{
                in = new float[n];
                out = new float[n];
                dLdx = new float[n];
            }

        }
        ~RELU()
        {
            if (mode == "gpu_optimized" || mode == "gpu_naive")
            {
                cudaFree(in_d);
                cudaFree(out_d);
                cudaFree(dLdx_d);
            }
            else
            {
                delete [] in;
                delete [] out;
                delete [] dLdx;
            }
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
            dim3 a(ceil((float)n/1024),1,1);
            dim3 b(1024,1,1);
            cudaMemcpy(in_d, x, n*sizeof(float), cudaMemcpyDeviceToDevice);
            relu_forward<<<a,b>>>(in_d, out_d, n);
        }
        void backward_gpu(float * dLdy_d)
        {
            dim3 a(ceil((float)n/1024));
            dim3 b(1024);
            relu_backward<<<a,b>>>(dLdy_d, in_d, dLdx_d, n);
        }

        void forward_cpu(float * x)
        {
            REP0(i,n)
            {
                in[i] = x[i];
                if(x[i] > 0 )out[i] = x[i];
                else out[i] = 0.0;
            }
        }

        void backward_cpu(float * dLdy)
        {
            REP0(i,n)
            {
                if(in[i]>0)dLdx[i] = dLdy[i] * 1;
                else dLdx[i] = 0.0;
            }
        }
};



#endif
