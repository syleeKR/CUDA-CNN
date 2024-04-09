using namespace std;
pair<vector<pair<vector<float>, int>>, vector<pair<vector<float>, int>>> read();
pair<vector<pair<vector<float>, int>>, vector<pair<vector<float>, int>>> get_training_data(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset);


pair<vector<pair<vector<float>, int>>, vector<pair<vector<float>, int>>> read()
{
    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("./mnist/");

    pair<vector<pair<vector<float>, int>>, vector<pair<vector<float>, int>>> data = get_training_data(dataset);
    return data;
}

vector<float> convert_to_float_vector(vector<uint8_t> & input)
{
    vector<float> tmp;
    REP0(i, sz(input))tmp.pb((float)input[i]/100);
    return tmp;
}

pair<vector<pair<vector<float>, int>>, vector<pair<vector<float>, int>>> get_training_data(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset)
{
    vector<pair<vector<float>, int>> train, test;
    
    REP0(i, sz(dataset.training_images))
    {
        pair<vector<float>, int> tmp;
        tmp.fi = convert_to_float_vector(dataset.training_images[i]);
        tmp.se = (int)dataset.training_labels[i];
        train.pb(tmp);
    }

    REP0(i, sz(dataset.test_images))
    {
        pair<vector<float>, int> tmp;
        tmp.fi = convert_to_float_vector(dataset.test_images[i]);
        tmp.se = (int)dataset.test_labels[i];
        test.pb(tmp);
    }
    return {train, test};
}

