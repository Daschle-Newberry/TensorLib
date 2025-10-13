cmake_minimum_required(VERSION 4.0)
project(TensorLib C)

set(CMAKE_C_STANDARD 11)

include_directories(include)
add_executable(TensorLib src/main.c
        include/tensor.h
        src/string_builder.c
        include/string_builder.h
        src/tensor.c
)
