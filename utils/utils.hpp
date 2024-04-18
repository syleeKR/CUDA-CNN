
#ifndef utils_hpp
#define utils_hpp

double duration(clock_t start, clock_t end)
{
    return (double)(end - start) / CLOCKS_PER_SEC;
}


float * make_image_batch(DATA &train_data, vint & shuffled_index, int iter, int batch_size)
{
    float * total = new float[batch_size * 28 * 28];
    int p = 0;
    for(int i=iter*batch_size; i<iter*batch_size+batch_size; i++)
    {
        for(float x : train_data[shuffled_index[i]].fi)
        {
            total[p] = x;
            p++;
        }
    }
    return total;
}
int * make_target_batch(DATA &train_data, vint & shuffled_index, int iter, int batch_size)
{
    int * total = new int[batch_size];
    int p = 0;
    for(int i=iter*batch_size; i<iter*batch_size+batch_size; i++)
    {
        total[p] = train_data[shuffled_index[i]].se;
        p++;
    }
    return total;
}

vint randomPermutation(int n) {
    // Create a vector containing the numbers 1 to n
    vint result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = i;
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine(seed);
    std::shuffle(result.begin(), result.end(), engine);

    return result;
}


void check_gpu_array(bool * target, int checksize, int step =100)
{
    bool * check = new bool[checksize];
    cudaMemcpy(check, target, sizeof(bool) * checksize, cudaMemcpyDeviceToHost);
    cout<<"checking"<<endl;
    for(int i =0 ; i<checksize; i+=step)cout<<check[i]<<" ";
    cout<<endl;
    delete [] check;
}
void check_gpu_array(float * target, int checksize, int step =100)
{
    float * check = new float[checksize];
    cudaMemcpy(check, target, sizeof(float) * checksize, cudaMemcpyDeviceToHost);
    cout<<"checking"<<endl;
    for(int i =0 ; i<checksize; i+=step)cout<<check[i]<<" ";
    cout<<endl;
    delete [] check;
}


string cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "UNKNOWN";
    }
}

void cublas_matmul(const float* A, const float* B, float* C, int m, int k, int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;


    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                n, m, k, &alpha, 
                                  B, n,
                                  A, k,
                                  &beta, 
                                  C, n);


    cublasDestroy(handle);
}

#endif