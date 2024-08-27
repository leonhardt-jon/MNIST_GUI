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
    model.load_weights_from_binary("cnn_weights_v2.bin");
    
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
