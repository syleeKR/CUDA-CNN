
#ifndef fcc_hpp
#define fcc_hpp

class FCC
{
    private:
        int in_dim;
        int out_dim;
        v2float A;
        vfloat b;



    public:
        FCC(int in_dim, int out_dim):in_dim(in_dim), out_dim(out_dim)
        {

        }
        vector<float> forward(vector<float> &x){return x;}
        vector<float> backward(vector<float> & ygrad){return ygrad;}
        void update(float lr){return;}

};

#endif

