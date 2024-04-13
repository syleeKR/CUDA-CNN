


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
        CNN(int in_dim, int out_dim, int filter_size, vint  input_size):in_dim(in_dim), out_dim(out_dim), filter_size(filter_size)
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
            int ntot =(out_dim * in_dim * filter_size * filter_size);
            float bound = sqrt(6.0/(float)ntot);
            std::uniform_real_distribution<float> dis(-bound, +bound); 
            REP0(i, out_dim * in_dim * filter_size * filter_size){filter[i] =dis(gen);}
                        
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
            REP0(i, out_dim *in_dim*filter_size * filter_size){
                filter[i] -= lr * dLdf[i];
            }
        }
        
        void print(string s)
        {
            if(s == "out"){
            for(int i=0; i<B*out_dim*Hout *Wout; i+=100)cout<<out[i]<<" ";
            cout<<endl;}
            if(s == "grad")
            {
                cout<< "dLdx"<<endl;
                for(int i ;i <B*in_dim*H*W; i+= 100)cout<<dLdx[i]<<" ";
                cout<<endl;
                cout<<"dLdf"<<endl;
                float maxval= 0.0;
                for(int i; i <out_dim * in_dim * filter_size * filter_size; i+=100){cout<<dLdf[i]<<" ";maxval = max(maxval, abs(dLdf[i]));}
                cout<<"maxval of gradient : " << maxval<<endl;
                cout<<endl;
            }
        }


};

#endif

