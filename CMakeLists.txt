# 1. Add the CMake minimum required version and project name
cmake_minimum_required(VERSION 3.10)

project(NeuralNetworks LANGUAGES CXX)

# 2. Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3. Add executable target
add_executable(main main.cpp value.cpp nn.cpp)