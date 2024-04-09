#ifndef softmax_hpp
#define softmax_hpp

class SOFTMAX
{
    public:
        int B,C,H,W;
        float * in = nullptr;
        float * out = nullptr;
        float * dLdx = nullptr;

        SOFTMAX(){}
        SOFTMAX(vint & input_size)
        {
            B = input_size[0];
            C = input_size[1];
            H = input_size[2];
            W = input_size[3];

            in = new float[B*C*H*W];
            out = new float[B*C*H*W];
            dLdx = new float[B*C*H*W];
        }
        ~SOFTMAX()
        {
            delete [] in;
            delete [] out;
            delete [] dLdx;
        }
        int index(int i, int j, int k=0, int l=0)
        {
            return C*H*W*i + H*W *j + W *k + l;
        }
        void forward(float * x)
        {
            REP0(i,B)
            {
                float total= 0.0;
                REP0(j, C)
                {
                    int p = index(i,j);
                    in[p] = x[p];
                    total += exp(x[p]);
                }
                REP0(j,C)
                {
                    int p = index(i,j);
                    out[p] = exp(x[p])/total;
                }
            }
        }

        void backward(float * dLdy)
        {
            REP0(i,B)
            {
                REP0(j, C)dLdx[index(i,j)] = 0.0; // zero out gradients
                
                REP0(ypointer, C)
                {
                    REP0(xpointer, C)
                    {
                        int p = index(i,xpointer);
                        if (xpointer==ypointer)
                        {
                            float grad = out[ypointer] * (1 - out[ypointer]);
                            dLdx[p]+=grad;
                        }
                        else
                        {
                            float grad = -out[ypointer] * out[xpointer];
                            dLdx[p] += grad;
                        }

                    }

                }
            }
        }
};



#endif
