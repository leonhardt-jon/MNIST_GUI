#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include "cnn.h"

//class CNN {
//private:
//	bool debug_mode = false; 
//
//	void debug_print(const std::string& message) {
//	    if (debug_mode) {
//		std::cout << "DEBUG: " << message << std::endl;
//	    }
//	}
//	// ReLU Activation
//	static float relu(float x) {
//		return std::max(0.0f, x); 
//	}
//
//    static std::vector<std::vector<std::vector<float>>> convolve2d(
//        const std::vector<std::vector<std::vector<float>>>& input, 
//        const std::vector<std::vector<std::vector<std::vector<float>>>>& kernel) {
//        
//        int input_channels = input.size();
//        int input_height = input[0].size();
//        int input_width = input[0][0].size();
//        int num_filters = kernel.size();
//        int kernel_height = kernel[0][0].size();
//        int kernel_width = kernel[0][0][0].size();
//        
//        // Calculate padding
//        int pad_height = (kernel_height - 1) / 2;
//        int pad_width = (kernel_width - 1) / 2;
//        
//        // Create padded input
//        std::vector<std::vector<std::vector<float>>> padded_input(
//            input_channels,
//            std::vector<std::vector<float>>(
//                input_height + 2 * pad_height,
//                std::vector<float>(input_width + 2 * pad_width, 0.0)
//            )
//        );
//        
//        // Fill padded input
//        for (int c = 0; c < input_channels; ++c) {
//            for (int i = 0; i < input_height; ++i) {
//                for (int j = 0; j < input_width; ++j) {
//                    padded_input[c][i + pad_height][j + pad_width] = input[c][i][j];
//                }
//            }
//        }
//        
//        // Output dimensions are now the same as input dimensions
//        int output_height = input_height;
//        int output_width = input_width;
//        
//        std::vector<std::vector<std::vector<float>>> output(
//            num_filters,
//            std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0.0))
//        );
//        
//        for (int d = 0; d < num_filters; ++d) {
//            for (int i = 0; i < output_height; ++i) {
//                for (int j = 0; j < output_width; ++j) {
//                    float sum = 0.0;
//                    for (int z = 0; z < input_channels; ++z) {
//                        for (int k = 0; k < kernel_height; ++k) {
//                            for (int l = 0; l < kernel_width; ++l) {
//                                sum += padded_input[z][i+k][j+l] * kernel[d][z][k][l];
//                            }
//                        }
//                    }
//                    output[d][i][j] = sum;
//                }
//            }
//        }
//        
//        return output;
//    }
//
//	
//	static std::vector<std::vector<std::vector<float>>> max_pool( const std::vector<std::vector<std::vector<float>>>& input, int pool_size){
//		int input_depth = input.size();
//		int input_height = input[0].size();
//		int input_width = input[0][0].size();
//
//		int output_depth = input_depth; 
//		int output_height = input_height / pool_size;
//		int output_width = input_width / pool_size;
//
//		std::vector<std::vector<std::vector<float>>> output(
//			output_depth, 
//			std::vector<std::vector<float>>(
//				output_height,
//				std::vector<float>(output_width, 0.0)
//			)
//		);
//		for (int d = 0; d < output_depth; ++d) {
//			for (int i = 0; i < output_height; ++i) {
//				for (int j = 0; j < output_width; ++j) {
//       					float max_val = -std::numeric_limits<float>::infinity();
//       					for (int k = 0; k < pool_size; ++k){
//       						for (int l = 0; l < pool_size; ++l) {
//       							max_val = std::max(max_val, input[d][i*pool_size+k][j*pool_size+l]);
//       						}
//       					}
//       					output[d][i][j] = max_val;
//				}
//			}
//		}
//		return output;
//	}
//
//	static std::vector<float> flatten(const std::vector<std::vector<std::vector<float>>> & input) {
//		std::vector<float> output;
//		for (const auto& layer : input){ 
//			for (const auto& row : layer) {
//       				output.insert(output.end(), row.begin(), row.end());
//       			}
//		}
//		return output;
//	}
//
//	static std::vector<float> fully_connected(
//		const std::vector<float>& input, 
//		const std::vector<std::vector<float>>& weights,
//		const std::vector<float>& biases){
//
//		std::vector<float> output(weights.size(), 0.0);
//		for (size_t i =0; i < weights.size(); ++i) {
//			for (size_t j = 0; j < input.size(); ++j) {
//       				output[i] += input[j] * weights[i][j];
//       			}
//		}
//		return output;
//	}
//
//	static std::vector<float> dropout(
//		const std::vector<float> input, 
//		float p = 0.25) {
//
//		std::vector<float> output = input; 
//		std::random_device rd; 
//		std::mt19937 gen(rd()); 
//		std::bernoulli_distribution d(1 - p); 
//		for (auto& val : output) {
//			if (!d(gen)) val = 0;
//			else val /= (1-p); 
//		}
//		return output;
//	}
//
//
//	std::vector<std::vector<std::vector<std::vector<float>>>> reshape_4d(const std::vector<float>& flat, int d1, int d2, int d3, int d4) {
//    	    std::vector<std::vector<std::vector<std::vector<float>>>> reshaped(d1, std::vector<std::vector<std::vector<float>>>(d2, std::vector<std::vector<float>>(d3, std::vector<float>(d4))));
//    	    int index = 0;
//    	    for (int i = 0; i < d1; ++i) {
//    	        for (int j = 0; j < d2; ++j) {
//    	            for (int k = 0; k < d3; ++k) {
//    	                for (int l = 0; l < d4; ++l) {
//    	                    reshaped[i][j][k][l] = flat[index++];
//    	                }
//    	            }
//    	        }
//    	    }
//    	    return reshaped;
//    	}
//
//	std::vector<std::vector<float>> reshape_2d(const std::vector<float>& flat, int d1, int d2) {
//    	    std::vector<std::vector<float>> reshaped(d1, std::vector<float>(d2));
//    	    int index = 0;
//    	    for (int i = 0; i < d1; ++i) {
//    	        for (int j = 0; j < d2; ++j) {
//    	            reshaped[i][j] = flat[index++];
//    	        }
//    	    }
//    	    return reshaped;
//    	}
//
//
//
//	std::vector<std::vector<std::vector<std::vector<float>>>> conv1_weights;
//	std::vector<std::vector<std::vector<std::vector<float>>>> conv2_weights;
//	std::vector<std::vector<float>> fc1_weights;
//	std::vector<std::vector<float>> fc2_weights;
//	std::vector<float> conv1_bias; 
//	std::vector<float> conv2_bias; 
//	std::vector<float> fc1_bias;
//	std::vector<float> fc2_bias;
//
//
//
//
//public: 
//        void load_weights_from_binary(const std::string& filename) {
//        debug_print("Attempting to open file: " + filename);
//        std::ifstream file(filename, std::ios::binary);
//        if (!file) {
//            throw std::runtime_error("Failed to open weights file: " + filename);
//        }
//        debug_print("File opened successfully");
//
//        while (file.peek() != EOF) {
//            // Read parameter name
//            uint32_t name_length;
//            file.read(reinterpret_cast<char*>(&name_length), sizeof(uint32_t));
//            std::string name(name_length, '\0');
//            file.read(&name[0], name_length);
//            debug_print("Reading weights for: " + name);
//
//            // Read dimensions
//            uint32_t num_dims;
//            file.read(reinterpret_cast<char*>(&num_dims), sizeof(uint32_t));
//            std::vector<uint32_t> dims(num_dims);
//            file.read(reinterpret_cast<char*>(dims.data()), num_dims * sizeof(uint32_t));
//            
//            std::string dims_str = "Dimensions: ";
//            for (auto d : dims) {
//                dims_str += std::to_string(d) + " ";
//            }
//            debug_print(dims_str);
//
//            // Calculate total number of elements
//            uint32_t total_elements = 1;
//            for (uint32_t dim : dims) {
//                total_elements *= dim;
//            }
//            debug_print("Total elements: " + std::to_string(total_elements));
//
//            // Read weights
//            std::vector<float> weights(total_elements);
//            file.read(reinterpret_cast<char*>(weights.data()), total_elements * sizeof(float));
//
//            if (weights.empty()) {
//                throw std::runtime_error("Failed to read weights for: " + name);
//            }
//
//            // Map weights to correct location
//            if (name == "conv1.weight") {
//                conv1_weights = reshape_4d(weights, dims[0], dims[1], dims[2], dims[3]);
//                debug_print("Loaded conv1_weights: " + std::to_string(conv1_weights.size()) + " x " + 
//                            std::to_string(conv1_weights[0].size()) + " x " + 
//                            std::to_string(conv1_weights[0][0].size()) + " x " + 
//                            std::to_string(conv1_weights[0][0][0].size()));
//            } else if (name == "conv1.bias") {
//                conv1_bias = weights;
//                debug_print("Loaded conv1_bias: " + std::to_string(conv1_bias.size()));
//            } else if (name == "conv2.weight") {
//                conv2_weights = reshape_4d(weights, dims[0], dims[1], dims[2], dims[3]);
//                debug_print("Loaded conv2_weights: " + std::to_string(conv2_weights.size()) + " x " + 
//                            std::to_string(conv2_weights[0].size()) + " x " + 
//                            std::to_string(conv2_weights[0][0].size()) + " x " + 
//                            std::to_string(conv2_weights[0][0][0].size()));
//            } else if (name == "conv2.bias") {
//                conv2_bias = weights;
//                debug_print("Loaded conv2_bias: " + std::to_string(conv2_bias.size()));
//            } else if (name == "fc1.weight") {
//                fc1_weights = reshape_2d(weights, dims[0], dims[1]);
//                debug_print("Loaded fc1_weights: " + std::to_string(fc1_weights.size()) + " x " + 
//                            std::to_string(fc1_weights[0].size()));
//            } else if (name == "fc1.bias") {
//                fc1_bias = weights;
//                debug_print("Loaded fc1_bias: " + std::to_string(fc1_bias.size()));
//            } else if (name == "fc2.weight") {
//                fc2_weights = reshape_2d(weights, dims[0], dims[1]);
//                debug_print("Loaded fc2_weights: " + std::to_string(fc2_weights.size()) + " x " + 
//                            std::to_string(fc2_weights[0].size()));
//            } else if (name == "fc2.bias") {
//                fc2_bias = weights;
//                debug_print("Loaded fc2_bias: " + std::to_string(fc2_bias.size()));
//            } else {
//                debug_print("Unknown weight name: " + name);
//            }
//        }
//
//        // Add error checking after loading all weights
//        if (conv1_weights.empty() || conv2_weights.empty() || fc1_weights.empty() || fc2_weights.empty()) {
//            throw std::runtime_error("Some weights were not loaded properly");
//        }
//        debug_print("All weights loaded successfully");
//    }
//
//
//    std::vector<float> forward(
//        const std::vector<std::vector<std::vector<float>>>& input)
//     {
//	
//        if (input.empty() || input[0].empty() || input[0][0].empty()) {
//            throw std::runtime_error("Input is empty or has incorrect dimensions");
//        }
//
//        if (conv1_weights.empty() || conv2_weights.empty() || fc1_weights.empty() || fc2_weights.empty()) {
//            throw std::runtime_error("Weights have not been loaded");
//        }
//
//        auto conv1_output = convolve2d(input, conv1_weights);
//	for (size_t i = 0; i < conv1_output.size(); ++i){
//	    for (auto& row : conv1_output[i]) {
//		for (auto& val : row) {
//		    val += relu(conv1_bias[i]);
//		}
//	    }
//	}
//        auto pool1_output = max_pool(conv1_output, 2);
//
//        auto conv2_output = convolve2d(pool1_output, conv2_weights);
//	for (size_t i = 0; i < conv2_output.size(); ++i){
//	    for (auto& row : conv2_output[i]) {
//		for (auto& val : row) {
//		    val += relu(conv2_bias[i]);
//		}
//	    }
//	}
//        auto pool2_output = max_pool(conv2_output, 2);
//
//        auto flattened = flatten(pool2_output);
//
//
//        auto fc1_output = fully_connected(flattened, fc1_weights, fc1_bias);
//        for (auto& val : fc1_output) {
//            val = relu(val);
//        }
//
//	auto fc2_output = fully_connected(fc1_output, fc2_weights, fc2_bias);
//        return fc2_output;
//    }
//};

const int CANVAS_SIZE = 28;
const int BRUSH_RADIUS = 2; 
const int BRUSH_STRENGTH = 100;
const int PIXEL_SIZE = 10;

const double CLASSIFICATION_INTERVAL = 0.5; // Classification interval in seconds
double last_classification_time = 0.0;
//std::vector<bool> canvas(CANVAS_SIZE * CANVAS_SIZE, false);
std::vector<unsigned char> canvas(CANVAS_SIZE * CANVAS_SIZE, 0);
CNN model;
std::vector<float> classification_probabilities(10, 0.0f);
std::array<float, 10> animated_probabilities = {0};

const float ANIMATION_SPEED = 0.1f;

void clear_canvas() {
    std::fill(canvas.begin(), canvas.end(), 0);
    std::fill(classification_probabilities.begin(), classification_probabilities.end(), 0.0f);
    std::fill(animated_probabilities.begin(), animated_probabilities.end(), 0.0f);
}

void update_animated_probabilities() {
    for (int i = 0; i < 10; ++i) {
        animated_probabilities[i] += (classification_probabilities[i] - animated_probabilities[i]) * ANIMATION_SPEED;
    }
}

ImU32 get_bar_color(int index, float alpha) {
    const std::array<ImVec4, 10> colors = {
        ImVec4(0.8f, 0.1f, 0.1f, alpha),  // Red
        ImVec4(0.1f, 0.8f, 0.1f, alpha),  // Green
        ImVec4(0.1f, 0.1f, 0.8f, alpha),  // Blue
        ImVec4(0.8f, 0.8f, 0.1f, alpha),  // Yellow
        ImVec4(0.8f, 0.1f, 0.8f, alpha),  // Magenta
        ImVec4(0.1f, 0.8f, 0.8f, alpha),  // Cyan
        ImVec4(0.9f, 0.5f, 0.1f, alpha),  // Orange
        ImVec4(0.5f, 0.1f, 0.9f, alpha),  // Purple
        ImVec4(0.1f, 0.5f, 0.3f, alpha),  // Teal
        ImVec4(0.5f, 0.3f, 0.1f, alpha)   // Brown
    };
    return ImGui::ColorConvertFloat4ToU32(colors[index]);
}

std::vector<std::vector<std::vector<float>>> canvas_to_input() {
    const float mean = 0.1307f;
    const float std_dev = 0.3081f;
    
    std::vector<std::vector<std::vector<float>>> input(1, std::vector<std::vector<float>>(CANVAS_SIZE, std::vector<float>(CANVAS_SIZE, 0.0f)));
    
    for (int y = 0; y < CANVAS_SIZE; ++y) {
        for (int x = 0; x < CANVAS_SIZE; ++x) {
            float pixel_value = canvas[y * CANVAS_SIZE + x] / 255.0f;
            input[0][y][x] = (pixel_value - mean) / std_dev;
        }
    }
    
    return input;
}

std::vector<float> normalize_output(const std::vector<float>& output) {
    std::vector<float> normalized(output.size()); 
    float max_val = *std::max_element(output.begin(), output.end()); 

    float sum = 0.0f; 
    for (size_t i = 0; i < output.size(); ++i) {
	normalized[i] = std::exp(output[i] - max_val); 
	sum += normalized[i];
    }

    for (float& val : normalized) {
	val /= sum; 
    }
    return normalized;
}

void classify() {
    auto input = canvas_to_input();
    auto raw_output = model.forward(input);
    auto normalized_output = normalize_output(raw_output);
    for (int i = 0; i < 10; ++i) {
        classification_probabilities[i] = normalized_output[i];
    }
}

void draw_canvas() {
    ImGui::SetNextWindowSize(ImVec2(700, 400), ImGuiCond_FirstUseEver);
    ImGui::Begin("MNIST Digit Classifier");
    
    ImGui::BeginChild("CanvasChild", ImVec2(300, 340), true);
    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
    ImVec2 canvas_sz = ImVec2(280, 280);
    ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
    
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
    
    ImGui::InvisibleButton("canvas", canvas_sz);
    
    if (ImGui::IsItemHovered()) {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        ImVec2 rel_pos = ImVec2(mouse_pos.x - canvas_p0.x, mouse_pos.y - canvas_p0.y);
        int center_x = static_cast<int>(rel_pos.x / 10);
        int center_y = static_cast<int>(rel_pos.y / 10);

        auto applyBrush = [&](int strength) {
            for (int y = std::max(0, center_y - BRUSH_RADIUS); y <= std::min(CANVAS_SIZE - 1, center_y + BRUSH_RADIUS); ++y) {
                for (int x = std::max(0, center_x - BRUSH_RADIUS); x <= std::min(CANVAS_SIZE - 1, center_x + BRUSH_RADIUS); ++x) {
                    float distance = std::sqrt(std::pow(x - center_x, 2) + std::pow(y - center_y, 2));
                    if (distance <= BRUSH_RADIUS) {
                        float intensity = 1.0f - (distance / BRUSH_RADIUS);
                        int index = y * CANVAS_SIZE + x;
                        int newValue = canvas[index] + static_cast<int>(strength * intensity);
                        canvas[index] = static_cast<unsigned char>(std::clamp(newValue, 0, 255));
                    }
                }
            }
        };
        
        if (ImGui::IsMouseDown(0) && center_x >= 0 && center_x < CANVAS_SIZE && center_y >= 0 && center_y < CANVAS_SIZE) {
            applyBrush(BRUSH_STRENGTH);
        }
        
        if (ImGui::IsMouseDown(1) && center_x >= 0 && center_x < CANVAS_SIZE && center_y >= 0 && center_y < CANVAS_SIZE) {
            applyBrush(-BRUSH_STRENGTH);
        }
    }

    for (int y = 0; y < CANVAS_SIZE; ++y) {
        for (int x = 0; x < CANVAS_SIZE; ++x) {
            unsigned char value = canvas[y * CANVAS_SIZE + x];
            ImVec2 p0 = ImVec2(canvas_p0.x + x * PIXEL_SIZE, canvas_p0.y + y * PIXEL_SIZE);
            ImVec2 p1 = ImVec2(p0.x + PIXEL_SIZE, p0.y + PIXEL_SIZE);
            draw_list->AddRectFilled(p0, p1, IM_COL32(value, value, value, 255));
        }
    }

    if (ImGui::Button("Clear Canvas")) {
        clear_canvas();
    }

    ImGui::SameLine();

    //if (ImGui::Button("Classify")) {
    //    auto input = canvas_to_input();
    //    auto output = model.forward(input);
    //    auto normalized_output = normalize_output(output); 
    //    for (int i = 0; i < 10; ++i) {
    //        classification_probabilities[i] = normalized_output[i];
    //    }
    //}

    ImGui::EndChild(); 

    ImGui::SameLine(); 

    ImGui::BeginChild("ChartChild", ImVec2(380, 340), true);
    
    update_animated_probabilities();

    ImGui::Text("Classification Probabilities:");
    ImVec2 graph_size(360, 200);
    
    ImDrawList* chart_draw_list = ImGui::GetWindowDrawList();
    ImVec2 chart_pos = ImGui::GetCursorScreenPos();
    ImVec2 chart_end = ImVec2(chart_pos.x + graph_size.x, chart_pos.y + graph_size.y);

    chart_draw_list->AddRectFilled(chart_pos, chart_end, IM_COL32(30, 30, 30, 255));

    float bar_width = graph_size.x / 10;
    for (int i = 0; i < 10; ++i) {
        float bar_height = animated_probabilities[i] * graph_size.y;
        ImVec2 bar_start(chart_pos.x + i * bar_width, chart_end.y);
        ImVec2 bar_end(bar_start.x + bar_width - 2, chart_end.y - bar_height);
        chart_draw_list->AddRectFilled(bar_start, bar_end, get_bar_color(i, 1.0f));

        // Add label
        char label[4];
        snprintf(label, sizeof(label), "%d", i);
        ImVec2 label_pos(bar_start.x + bar_width / 2 - ImGui::CalcTextSize(label).x / 2, chart_end.y + 5);
        chart_draw_list->AddText(label_pos, IM_COL32(200, 200, 200, 255), label);
    }

    ImGui::Dummy(graph_size);  // Reserve space for the chart

    auto max_it = std::max_element(animated_probabilities.begin(), animated_probabilities.end());
    int classification_result = std::distance(animated_probabilities.begin(), max_it);
    ImGui::Dummy(ImVec2(0.0f, 20.0f));
    ImGui::Text("Classification Result: %d", classification_result);
    ImGui::Text("Probability: %.2f%%", *max_it * 100);

    ImGui::EndChild();
    
    ImGui::End();
    
}


int main() {
    if (!glfwInit())
        return 1;
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "Digit Classifier", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return 1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Load CNN weights
    model.load_weights_from_binary("../cnn_weights_v2.bin");
    
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
	
	double current_time = glfwGetTime(); 
	if (current_time - last_classification_time >= CLASSIFICATION_INTERVAL) {
	    classify();
	    last_classification_time = current_time;
	}
        
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        draw_canvas();
        
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}
