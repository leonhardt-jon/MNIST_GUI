# MNIST_GUI


[![Classification Demo](https://raw.githubusercontent.com/leonhardt-jon/MNIST_GUI/main/MNIST_Classifier_thumbnail.jpg)](https://raw.githubusercontent.com/leonhardt-jon/MNIST_GUI/main/MNIST_Classifier.mp4)


(Play video!^^^)

Draw MNIST digits and classify in real time!
Fun weekend project and learning experience. Full scope of project:
1. Trained CNN in pytorch on MNIST images
2. Custom binary dataloader to load weights (Just felt like it / wanted to reduce dependencies)
3. Implemented CNN forward pass in C++ (cnn.cpp)
4. Built GUI using dearImgui

## Dataset
https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data

## Compiling 

```console
git clone https://github.com/ocornut/imgui.git
```

```console
g++ -o real_time_MNIST gui.cpp cnn.cpp imgui/*.cpp imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp -I./imgui -I./imgui
/backends  -lGL -lglfw
```
