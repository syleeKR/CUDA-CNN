
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


#include <bits/stdc++.h>
#include "./mnist/mnist_reader.hpp"
#include "./mnist/read.hpp"

#include "relu.hpp"
#include "maxpool.hpp"
#include "cnn.hpp"
#include "softmax.hpp"

#include "convnet.hpp"

using namespace std;

float * make_image_batch(DATA &train_data, int start, int len)
{
    vfloat total;
    for(int i=start; i<start+len; i++)
    {
        vfloat tmp = train_data[i].fi;
        total.insert(total.end(), tmp.begin(), tmp.end());
    }
    return total.data();
}
void make_target_batch(DATA &train_data, int start, int len, int * targets)
{
    vint total;
    for(int i=start; i<start+len; i++)
    {
        targets[i-start] = train_data[i].se;
    }
}
int main() {

    pair<DATA,DATA> data =  read();
    DATA train_data = data.fi; DATA test_data = data.se;
    int n_train = sz(train_data); int n_test = sz(test_data);

    int batch = 1;
    ConvNet net = ConvNet(batch);
    int total_epoch = 100;
    float lr = 0.001;

    REP0(epoch, total_epoch)
    {
        float average_loss = 0;

        for(int i=0 ; i<n_train; i+=batch)
        {
            float * image = make_image_batch(train_data, i, min(batch, n_train-i));
            int * targets = new int[min(batch, n_train-i)];
            make_target_batch(train_data, i, min(batch, n_train-i), targets);

            float * y = net.forward(image);
            float loss = net.cross_entropy_loss(y, targets);
            net.backward(y, targets);
            net.update(lr);
            if(i<1000)average_loss += loss;
            else if (i==1000)average_loss /= i;
            else average_loss = average_loss *0.99 + loss*0.01;
            if(i>= 1000 && (i%1000==0))cout<<"iteration "<<i<<" average_loss "<< average_loss<<endl;
            delete[] targets;
        }

        // eval
        int correct = 0;
        REP0(i , n_test)
        {
            float * image = test_data[i].fi.data();
            int target = test_data[i].se;

            if(net.predict(image)==target)correct+=1;
        }
        cout<< "epoch "<< epoch<<"accuracy " << ((float) correct / n_test)<<endl; 
    }

}