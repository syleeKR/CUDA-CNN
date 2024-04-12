#ifndef convnet_hpp
#define convnet_hpp

class ConvNet
{
    
    public:
        int B;
        CNN cnn1;
        MAXPOOL maxpool1;
        RELU relu1;
        CNN cnn2;
        MAXPOOL maxpool2;
        RELU relu2;
        CNN cnn3;
        SOFTMAX softmax;

        ConvNet(int B):B(B){
            vint input_size;
            // B 1 28 28
            input_size = {B,1,28,28};
            cnn1 = CNN(1, 32, 5, input_size);
            // B 32 24 24
            input_size = {B,32,24,24};
            maxpool1 = MAXPOOL(input_size);
            // B 32 12 12
            relu1 = RELU(B * 32 * 12 * 12);
            // B 32 12 12
            input_size = {B,32,12,12};
            cnn2 = CNN(32, 64, 3, input_size);
            // B 64 10 10
            input_size = {B,64,10,10};
            maxpool2 = MAXPOOL(input_size);
            // B 64 5 5
            relu2 = RELU(B * 64 * 5 * 5);
            // B 64 5 5
            input_size = {B,64,5,5};
            cnn3 = CNN(64, 10, 5, input_size);
            // B 10 1 1
            input_size = {B,10,1,1};
            softmax = SOFTMAX(input_size);
        }
        float cross_entropy_loss(float * y, int y_true)
        {
            return -log(y[y_true]);
        }

        float * forward(float * x0)
        {
            cnn1.forward(x0);
            maxpool1.forward(cnn1.out);
            relu1.forward(maxpool1.out);
  
            cnn2.forward(relu1.out);
            maxpool2.forward(cnn2.out);
            relu2.forward(maxpool2.out);
            cnn3.forward(relu2.out);
            
            softmax.forward(cnn3.out);
            return softmax.out;
        }

        void backward(float * y, int ytrue)
        {
            float * dL_over_dy = new float[10];
            REP0(i, 10)dL_over_dy[i] = 0.0;
            dL_over_dy[ytrue] = -1.0 / y[ytrue];

            softmax.backward(dL_over_dy);
            cnn3.backward(softmax.dLdx);
            relu2.backward(cnn3.dLdx);
            maxpool2.backward(relu2.dLdx);
            cnn2.backward(maxpool2.dLdx);
            relu1.backward(cnn2.dLdx);
            maxpool1.backward(relu1.dLdx);
            cnn1.backward(maxpool1.dLdx);

            delete [] dL_over_dy;
        }

        void update(float lr)
        {
            cnn1.update(lr);
            cnn2.update(lr);
            cnn3.update(lr);
        }

        int predict(float * x)
        {
            float * y = forward(x);

            int maxv = -1;
            int maxindex = 0;
            REP0(i, 10)
            {
                if (y[i] > maxv)
                {
                    maxv = y[i];
                    maxindex = i;
                }
            }
            return maxindex;
        }
};


#endif