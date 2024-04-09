

#ifndef relu_hpp
#define relu_hpp

class RELU
{
    public:
        int n;
        float * in = nullptr;
        float * out = nullptr;
        float * dLdx = nullptr;

        RELU(){}
        RELU(int n):n(n)
        {
            in = new float[n];
            out = new float[n];
            dLdx = new float[n];
        }
        ~RELU()
        {
            delete [] in;
            delete [] out;
            delete [] dLdx;
        }
        void forward(float * x)
        {
            REP0(i,n)
            {
                in[i] = x[i];
                if(x[i] > 0 )out[i] = x[i];
                else out[i] = 0;
            }
        }

        void backward(float * dLdy)
        {
            REP0(i,n)
            {
                if(in[i]>0)dLdx[i] = dLdy[i] * 1;
                else dLdx[i] = 0;
            }
        }
};



#endif
