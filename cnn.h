#ifndef CNN_H
#define CNN_H

#include <vector>
#include <string>

class CNN {
private:
    bool debug_mode;

    void debug_print(const std::string& message);
    static float relu(float x);
    static std::vector<std::vector<std::vector<float>>> convolve2d(
        const std::vector<std::vector<std::vector<float>>>& input,
        const std::vector<std::vector<std::vector<std::vector<float>>>>& kernel);
    static std::vector<std::vector<std::vector<float>>> max_pool(
        const std::vector<std::vector<std::vector<float>>>& input, int pool_size);
    static std::vector<float> flatten(const std::vector<std::vector<std::vector<float>>>& input);
    static std::vector<float> fully_connected(
        const std::vector<float>& input,
        const std::vector<std::vector<float>>& weights,
        const std::vector<float>& biases);
    static std::vector<float> dropout(const std::vector<float> input, float p);

    std::vector<std::vector<std::vector<std::vector<float>>>> reshape_4d(
        const std::vector<float>& flat, int d1, int d2, int d3, int d4);
    std::vector<std::vector<float>> reshape_2d(const std::vector<float>& flat, int d1, int d2);

    std::vector<std::vector<std::vector<std::vector<float>>>> conv1_weights;
    std::vector<std::vector<std::vector<std::vector<float>>>> conv2_weights;
    std::vector<std::vector<float>> fc1_weights;
    std::vector<std::vector<float>> fc2_weights;
    std::vector<float> conv1_bias;
    std::vector<float> conv2_bias;
    std::vector<float> fc1_bias;
    std::vector<float> fc2_bias;

public:
    void load_weights_from_binary(const std::string& filename);
    std::vector<float> forward(const std::vector<std::vector<std::vector<float>>>& input);
};

#endif // CNN_H
