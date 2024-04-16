
#define REP0(i,n) for(int i=0; i<n; i++)
#define pb push_back
#define fi first
#define se second
#define vfloat vector<float>
#define v2float vector<vector<float>>
#define v3float vector<vector<vector<float>>>
#define v4float vector<vector<vector<vector<float>>>>
#define DATA vector<pair<vector<float>, int>>
#define sz(v) ((int)(v).size())
#define vint vector<int>


#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <random>
#include <chrono>
#include <string>
#include <ctime>
#include <algorithm>

#include "./mnist/mnist_reader.hpp"
#include "./mnist/read.hpp"

#include "./gpu_kernels/convolution_direct.cuh"
#include "./gpu_kernels/convolution_matmul.cuh"

#include "./networks/relu.hpp"
#include "./networks/maxpool.hpp"
#include "./networks/cnn.hpp"
#include "./networks/softmax.hpp"
#include "./networks/convnet.hpp"

#include "./utils/utils.hpp"

using namespace std;


void train(DATA & train_data, vint & shuffled_index, int batch_size, ConvNet & net, float lr)
{
    float average_loss = 0;
    int n_train = sz(train_data);
    int total_iteration = n_train / batch_size;
    REP0(iter, total_iteration)
    {
        float * images = make_image_batch(train_data, shuffled_index, iter, batch_size);
        int * targets = make_target_batch(train_data, shuffled_index, iter, batch_size);

        float * y = net.forward(images);
        float loss = net.cross_entropy_loss(y, targets);
        net.backward(y, targets);
        net.update(lr);

        delete[] images;
        delete[] targets;
        if(iter==0)average_loss = loss;
        else average_loss = average_loss *0.99 + loss*0.01;
        if((iter%100)==0)cout<<"iteration "<<iter<<" / "<< total_iteration<<" average_loss "<< average_loss<<endl;
    }

}
void eval(DATA & test_data, int batch_size, ConvNet & net, int epoch)
{
    int correct = 0;
    int n_test = sz(test_data);
    int total_iteration = n_test/batch_size;
    vint ordinary_index(n_test); REP0(i,n_test)ordinary_index[i]=i;
    REP0(iter , total_iteration)
    {
        float * images = make_image_batch(test_data, ordinary_index, iter, batch_size);
        int * targets = make_target_batch(test_data, ordinary_index, iter, batch_size);

        int * predictions = net.predict(images);
        REP0(b, batch_size){if(predictions[b] == targets[b])correct++;}
        
        delete [] images;
        delete [] targets;
        delete [] predictions;
    }
    cout<< "epoch "<< epoch<<" accuracy " << ((float) correct / n_test)<<endl; 

}
int main(int args, char * argv[])
{
    // read device mode
    string device = "cpu";
    if(args>1){string mode = argv[1]; if(mode=="gpu")device = "gpu"; if(mode=="gpu_mm")device = "gpu_mm";}

    // prepare Mnist data
    pair<DATA,DATA> data =  read();
    DATA train_data = data.fi; DATA test_data = data.se;

    // set up some hyperparameters and network
    int batch_size = 64;
    ConvNet net = ConvNet(device , batch_size);
    int total_epoch = 3;
    float lr = 0.005;

    // run code
    REP0(epoch, total_epoch)
    {
        vint shuffled_index = randomPermutation(sz(train_data));
        clock_t start = clock();
        train(train_data, shuffled_index, batch_size, net, lr);
        clock_t end = clock();
        cout<<"train time : "<<duration(start, end)<<endl;
        eval(test_data, batch_size, net, epoch);
    }

}