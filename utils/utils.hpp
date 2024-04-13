

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