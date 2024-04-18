#ifndef convnet_hpp
#define convnet_hpp

class ConvNet
{
    
    public:
        int B;
        string mode;
        CNN * cnn1= nullptr;
        MAXPOOL * maxpool1= nullptr;
        RELU * relu1= nullptr;
        CNN * cnn2= nullptr;
        MAXPOOL * maxpool2= nullptr;
        RELU * relu2= nullptr;
        CNN * cnn3= nullptr;
        SOFTMAX * softmax = nullptr;
        float * x0_d = nullptr;
        float * dLdx_d = nullptr;


        ~ConvNet()
        {
            delete cnn1;
            delete cnn2;
            delete cnn3;
            delete maxpool1;
            delete maxpool2;
            delete relu1;
            delete relu2;
            delete softmax;
            if (mode=="gpu_optimized" || mode == "gpu_naive")
            {
                cudaFree(x0_d);
                cudaFree(dLdx_d);
            }

        }
        ConvNet(string mode, int B):mode(mode),B(B){
            if (mode=="gpu_optimized" || mode == "gpu_naive"){
            cudaMalloc((void **)&x0_d, sizeof(float)*B*1*28*28);
            cudaMalloc((void **)&dLdx_d, sizeof(float)*B*10);
            }

            vint input_size;

            input_size = {B,1,28,28};
            cnn1 = new CNN(1, 16, 5, input_size, mode);

            input_size = {B,16,24,24};
            maxpool1 = new MAXPOOL(input_size,mode);

            relu1 = new RELU(B * 16 * 12 * 12,mode);

            input_size = {B,16,12,12};
            cnn2 = new CNN(16, 32, 3, input_size, mode);

            input_size = {B,32,10,10};
            maxpool2 = new MAXPOOL(input_size,mode);

            relu2 = new RELU(B * 32 * 5 * 5,mode);

            input_size = {B,32,5,5};
            cnn3 = new CNN(32, 10, 5, input_size, mode);
            
            input_size = {B,10,1,1};
            softmax = new SOFTMAX(input_size);
        }
        float cross_entropy_loss(float * y, int * y_true)
        {
            float sum = 0.0;
            REP0(b, B)sum += -log(y[10 * b + y_true[b]] + 1e-8);
            return sum/B;
        }
        float * forward(float * x0)
        {
            if(mode == "gpu_optimized" || mode == "gpu_naive")return forward_gpu(x0);
            else return forward_cpu(x0);
        }
        void backward(float * y, int * targets)
        {
            if(mode == "gpu_optimized" || mode == "gpu_naive")backward_gpu(y, targets);
            else backward_cpu(y, targets);
        }

        float * forward_gpu(float * x0)
        {
            cudaMemcpy(x0_d, x0,  sizeof(float)*B*1*28*28, cudaMemcpyHostToDevice);

            cnn1->forward(x0_d);
            maxpool1->forward(cnn1->out_d);
            relu1->forward(maxpool1->out_d);
            cnn2->forward(relu1->out_d);         
            maxpool2->forward(cnn2->out_d);
            relu2->forward(maxpool2->out_d);
            cnn3->forward(relu2->out_d);

            float * out = new float[B *10];
            cudaMemcpy(out , cnn3->out_d, sizeof(float) * B * 10, cudaMemcpyDeviceToHost);
            softmax->forward(out);
            delete [] out;

            return softmax->out;
        }

        void backward_gpu(float * y, int * targets)
        {   
            
            float * dL_over_dy = new float[B * 10];
            REP0(i, B){
                REP0(j, 10)dL_over_dy[i*10 + j] = 0.0;
                dL_over_dy[i*10 + targets[i]] = -1.0 / max(y[i*10 + targets[i]] , (float)0.00001);
            }            
            softmax->backward(dL_over_dy);
            delete [] dL_over_dy;
            cudaMemcpy(dLdx_d, softmax->dLdx,  sizeof(float)*B*10, cudaMemcpyHostToDevice);

            cnn3->backward(dLdx_d);
            relu2->backward(cnn3->dLdx_d);
            maxpool2->backward(relu2->dLdx_d);
            cnn2->backward(maxpool2->dLdx_d);
            relu1->backward(cnn2->dLdx_d);
            maxpool1->backward(relu1->dLdx_d);
            cnn1->backward(maxpool1->dLdx_d);
        }



        float * forward_cpu(float * x0)
        {
            cnn1->forward(x0);
            maxpool1->forward(cnn1->out);
            relu1->forward(maxpool1->out);

  
            cnn2->forward(relu1->out);         
            maxpool2->forward(cnn2->out);
            relu2->forward(maxpool2->out);
            cnn3->forward(relu2->out);
            softmax->forward(cnn3->out);
            
            return softmax->out;
        }

        void backward_cpu(float * y, int * targets)
        {   
            float * dL_over_dy = new float[B * 10];
            REP0(i, B){
                REP0(j, 10)dL_over_dy[i*10 + j] = 0.0;
                dL_over_dy[i*10 + targets[i]] = -1.0 / max(y[i*10 + targets[i]] , (float)0.00001);
            }            
            
            softmax->backward(dL_over_dy);            
            cnn3->backward(softmax->dLdx);

            relu2->backward(cnn3->dLdx);
            maxpool2->backward(relu2->dLdx);
            cnn2->backward(maxpool2->dLdx);

            relu1->backward(cnn2->dLdx);
            maxpool1->backward(relu1->dLdx);
            cnn1->backward(maxpool1->dLdx);
            delete [] dL_over_dy;
            
        }

        void update(float lr)
        {
            cnn1->update(lr);
            cnn2->update(lr);
            cnn3->update(lr);
        }

        int * predict(float * x)
        {
            float * y = forward(x);
            int * predictions = new int[B];

            REP0(b, B){
                float maxv = -1.0;
                int maxindex = 0;
                REP0(i, 10)
                {
                    if (y[10 * b + i] > maxv)
                    {
                        maxv = y[10 * b + i];
                        maxindex = i;
                    }
                }
                predictions[b] = maxindex;
            }
            return predictions;
        }
};


#endif