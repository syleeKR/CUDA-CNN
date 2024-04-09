


#ifndef cnn_hpp
#define cnn_hpp

class CNN
{
    public:
        int B,C,H,W;
        int Hout, Wout;
        int in_dim, out_dim, filter_size;


        float * in = nullptr;
        float * out = nullptr;
        float * dLdx = nullptr;

        float * filter =nullptr;
        float * dLdf = nullptr;

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
        CNN(int in_dim, int out_dim, int filter_size, vint & input_size):in_dim(in_dim), out_dim(out_dim), filter_size(filter_size)
        {
            B = input_size[0];
            C = input_size[1];
            H = input_size[2];
            W = input_size[3];
            Hout = H - 2 * (filter_size/2);
            Wout = W - 2 * (filter_size/2);

            in = new float[B*in_dim*H*W];
            out = new float[B*out_dim*Hout *Wout];
            dLdx = new float[B*C*H*W];
            filter = new float[out_dim * in_dim * filter_size * filter_size];
            dLdf = new float[out_dim * in_dim * filter_size * filter_size];
            
            std::random_device rd;  
            std::mt19937 gen(rd()); 
            std::uniform_real_distribution<float> dis(-0.001, +0.001); 
            
            REP0(i, out_dim)
            {
                REP0(j, in_dim)
                {
                    REP0(k1, filter_size)
                    {
                        REP0(k2, filter_size)
                        {
                            filter[index_filter(i,j,k1,k2)] =dis(gen);
                        }

                    }
                }
            }
        }
        ~CNN()
        {
            delete [] in;
            delete [] out;
            delete [] dLdx;
            delete [] filter;
            delete [] dLdf;
        }
        void forward(float * x)
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
            REP0(i,B){REP0(j,in_dim){REP0(k,H){REP0(l,W){in[index_in(i,j,k,l)] = x[index_in(i,j,k,l)];}}}}
        }

        void backward(float * dLdy)
        {
            return;
        }

        void update(float lr)
        {
            REP0(i, out_dim *in_dim*filter_size * filter_size){filter[i] -= lr * dLdf[i];}
        }


};

#endif

