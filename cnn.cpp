#include "cnn.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>

void CNN::debug_print(const std::string& message) {
    if (debug_mode) {
	std::cout << "DEBUG: " << message << std::endl;
    }
}
	// ReLU Activation
float CNN::relu(float x) {
    return std::max(0.0f, x); 
}

std::vector<std::vector<std::vector<float>>> CNN::convolve2d(
        const std::vector<std::vector<std::vector<float>>>& input, 
        const std::vector<std::vector<std::vector<std::vector<float>>>>& kernel) {
        
        int input_channels = input.size();
        int input_height = input[0].size();
        int input_width = input[0][0].size();
        int num_filters = kernel.size();
        int kernel_height = kernel[0][0].size();
        int kernel_width = kernel[0][0][0].size();
        
        int pad_height = (kernel_height - 1) / 2;
        int pad_width = (kernel_width - 1) / 2;
        
        std::vector<std::vector<std::vector<float>>> padded_input(
            input_channels,
            std::vector<std::vector<float>>(
                input_height + 2 * pad_height,
                std::vector<float>(input_width + 2 * pad_width, 0.0)
            )
        );
        
        for (int c = 0; c < input_channels; ++c) {
            for (int i = 0; i < input_height; ++i) {
                for (int j = 0; j < input_width; ++j) {
                    padded_input[c][i + pad_height][j + pad_width] = input[c][i][j];
                }
            }
        }
        
        int output_height = input_height;
        int output_width = input_width;
        
        std::vector<std::vector<std::vector<float>>> output(
            num_filters,
            std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0.0))
        );
        
        for (int d = 0; d < num_filters; ++d) {
            for (int i = 0; i < output_height; ++i) {
                for (int j = 0; j < output_width; ++j) {
                    float sum = 0.0;
                    for (int z = 0; z < input_channels; ++z) {
                        for (int k = 0; k < kernel_height; ++k) {
                            for (int l = 0; l < kernel_width; ++l) {
                                sum += padded_input[z][i+k][j+l] * kernel[d][z][k][l];
                            }
                        }
                    }
                    output[d][i][j] = sum;
                }
            }
        }
        
        return output;
    }

	
std::vector<std::vector<std::vector<float>>> CNN::max_pool( const std::vector<std::vector<std::vector<float>>>& input, int pool_size){
		int input_depth = input.size();
		int input_height = input[0].size();
		int input_width = input[0][0].size();

		int output_depth = input_depth; 
		int output_height = input_height / pool_size;
		int output_width = input_width / pool_size;

		std::vector<std::vector<std::vector<float>>> output(
			output_depth, 
			std::vector<std::vector<float>>(
				output_height,
				std::vector<float>(output_width, 0.0)
			)
		);
		for (int d = 0; d < output_depth; ++d) {
			for (int i = 0; i < output_height; ++i) {
				for (int j = 0; j < output_width; ++j) {
       					float max_val = -std::numeric_limits<float>::infinity();
       					for (int k = 0; k < pool_size; ++k){
       						for (int l = 0; l < pool_size; ++l) {
       							max_val = std::max(max_val, input[d][i*pool_size+k][j*pool_size+l]);
       						}
       					}
       					output[d][i][j] = max_val;
				}
			}
		}
		return output;
	}

std::vector<float> CNN::flatten(const std::vector<std::vector<std::vector<float>>> & input) {
		std::vector<float> output;
		for (const auto& layer : input){ 
			for (const auto& row : layer) {
       				output.insert(output.end(), row.begin(), row.end());
       			}
		}
		return output;
	}

std::vector<float> CNN::fully_connected(
		const std::vector<float>& input, 
		const std::vector<std::vector<float>>& weights,
		const std::vector<float>& biases){

		std::vector<float> output(weights.size(), 0.0);
		for (size_t i =0; i < weights.size(); ++i) {
			for (size_t j = 0; j < input.size(); ++j) {
       				output[i] += input[j] * weights[i][j];
       			}
		}
		return output;
	}

std::vector<float> CNN::dropout(
		const std::vector<float> input, 
		float p = 0.25) {

		std::vector<float> output = input; 
		std::random_device rd; 
		std::mt19937 gen(rd()); 
		std::bernoulli_distribution d(1 - p); 
		for (auto& val : output) {
			if (!d(gen)) val = 0;
			else val /= (1-p); 
		}
		return output;
	}


	std::vector<std::vector<std::vector<std::vector<float>>>> CNN::reshape_4d(const std::vector<float>& flat, int d1, int d2, int d3, int d4) {
    	    std::vector<std::vector<std::vector<std::vector<float>>>> reshaped(d1, std::vector<std::vector<std::vector<float>>>(d2, std::vector<std::vector<float>>(d3, std::vector<float>(d4))));
    	    int index = 0;
    	    for (int i = 0; i < d1; ++i) {
    	        for (int j = 0; j < d2; ++j) {
    	            for (int k = 0; k < d3; ++k) {
    	                for (int l = 0; l < d4; ++l) {
    	                    reshaped[i][j][k][l] = flat[index++];
    	                }
    	            }
    	        }
    	    }
    	    return reshaped;
    	}

	std::vector<std::vector<float>> CNN::reshape_2d(const std::vector<float>& flat, int d1, int d2) {
    	    std::vector<std::vector<float>> reshaped(d1, std::vector<float>(d2));
    	    int index = 0;
    	    for (int i = 0; i < d1; ++i) {
    	        for (int j = 0; j < d2; ++j) {
    	            reshaped[i][j] = flat[index++];
    	        }
    	    }
    	    return reshaped;
    	}



	//std::vector<std::vector<std::vector<std::vector<float>>>> conv1_weights;
	//std::vector<std::vector<std::vector<std::vector<float>>>> conv2_weights;
	//std::vector<std::vector<float>> fc1_weights;
	//std::vector<std::vector<float>> fc2_weights;
	//std::vector<float> conv1_bias; 
	//std::vector<float> conv2_bias; 
	//std::vector<float> fc1_bias;
	//std::vector<float> fc2_bias;




        void CNN::load_weights_from_binary(const std::string& filename) {
        debug_print("Attempting to open file: " + filename);
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open weights file: " + filename);
        }
        debug_print("File opened successfully");

        while (file.peek() != EOF) {
            uint32_t name_length;
            file.read(reinterpret_cast<char*>(&name_length), sizeof(uint32_t));
            std::string name(name_length, '\0');
            file.read(&name[0], name_length);
            debug_print("Reading weights for: " + name);

            uint32_t num_dims;
            file.read(reinterpret_cast<char*>(&num_dims), sizeof(uint32_t));
            std::vector<uint32_t> dims(num_dims);
            file.read(reinterpret_cast<char*>(dims.data()), num_dims * sizeof(uint32_t));
            
            std::string dims_str = "Dimensions: ";
            for (auto d : dims) {
                dims_str += std::to_string(d) + " ";
            }
            debug_print(dims_str);

            uint32_t total_elements = 1;
            for (uint32_t dim : dims) {
                total_elements *= dim;
            }
            debug_print("Total elements: " + std::to_string(total_elements));

            std::vector<float> weights(total_elements);
            file.read(reinterpret_cast<char*>(weights.data()), total_elements * sizeof(float));

            if (weights.empty()) {
                throw std::runtime_error("Failed to read weights for: " + name);
            }

            if (name == "conv1.weight") {
                conv1_weights = reshape_4d(weights, dims[0], dims[1], dims[2], dims[3]);
                debug_print("Loaded conv1_weights: " + std::to_string(conv1_weights.size()) + " x " + 
                            std::to_string(conv1_weights[0].size()) + " x " + 
                            std::to_string(conv1_weights[0][0].size()) + " x " + 
                            std::to_string(conv1_weights[0][0][0].size()));
            } else if (name == "conv1.bias") {
                conv1_bias = weights;
                debug_print("Loaded conv1_bias: " + std::to_string(conv1_bias.size()));
            } else if (name == "conv2.weight") {
                conv2_weights = reshape_4d(weights, dims[0], dims[1], dims[2], dims[3]);
                debug_print("Loaded conv2_weights: " + std::to_string(conv2_weights.size()) + " x " + 
                            std::to_string(conv2_weights[0].size()) + " x " + 
                            std::to_string(conv2_weights[0][0].size()) + " x " + 
                            std::to_string(conv2_weights[0][0][0].size()));
            } else if (name == "conv2.bias") {
                conv2_bias = weights;
                debug_print("Loaded conv2_bias: " + std::to_string(conv2_bias.size()));
            } else if (name == "fc1.weight") {
                fc1_weights = reshape_2d(weights, dims[0], dims[1]);
                debug_print("Loaded fc1_weights: " + std::to_string(fc1_weights.size()) + " x " + 
                            std::to_string(fc1_weights[0].size()));
            } else if (name == "fc1.bias") {
                fc1_bias = weights;
                debug_print("Loaded fc1_bias: " + std::to_string(fc1_bias.size()));
            } else if (name == "fc2.weight") {
                fc2_weights = reshape_2d(weights, dims[0], dims[1]);
                debug_print("Loaded fc2_weights: " + std::to_string(fc2_weights.size()) + " x " + 
                            std::to_string(fc2_weights[0].size()));
            } else if (name == "fc2.bias") {
                fc2_bias = weights;
                debug_print("Loaded fc2_bias: " + std::to_string(fc2_bias.size()));
            } else {
                debug_print("Unknown weight name: " + name);
            }
        }

        if (conv1_weights.empty() || conv2_weights.empty() || fc1_weights.empty() || fc2_weights.empty()) {
            throw std::runtime_error("Some weights were not loaded properly");
        }
        debug_print("All weights loaded successfully");
    }


    std::vector<float> CNN::forward(
        const std::vector<std::vector<std::vector<float>>>& input)
     {
	
        if (input.empty() || input[0].empty() || input[0][0].empty()) {
            throw std::runtime_error("Input is empty or has incorrect dimensions");
        }

        if (conv1_weights.empty() || conv2_weights.empty() || fc1_weights.empty() || fc2_weights.empty()) {
            throw std::runtime_error("Weights have not been loaded");
        }

        auto conv1_output = convolve2d(input, conv1_weights);
	for (size_t i = 0; i < conv1_output.size(); ++i){
	    for (auto& row : conv1_output[i]) {
		for (auto& val : row) {
		    val += relu(conv1_bias[i]);
		}
	    }
	}
        auto pool1_output = max_pool(conv1_output, 2);

        auto conv2_output = convolve2d(pool1_output, conv2_weights);
	for (size_t i = 0; i < conv2_output.size(); ++i){
	    for (auto& row : conv2_output[i]) {
		for (auto& val : row) {
		    val += relu(conv2_bias[i]);
		}
	    }
	}
        auto pool2_output = max_pool(conv2_output, 2);

        auto flattened = flatten(pool2_output);


        auto fc1_output = fully_connected(flattened, fc1_weights, fc1_bias);
        for (auto& val : fc1_output) {
            val = relu(val);
        }

	auto fc2_output = fully_connected(fc1_output, fc2_weights, fc2_bias);
        return fc2_output;
    }
