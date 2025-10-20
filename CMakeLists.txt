cmake_minimum_required(VERSION 4.0)

project(TensorLib C)

set(CMAKE_C_STANDARD 11)

add_library(TensorLib)
target_sources(TensorLib
        PRIVATE
            src/tensor.c
            src/string_builder.c
        PUBLIC
            FILE_SET HEADERS
            BASE_DIRS
                include
            FILES
                include/tensor.h
)

add_executable(TensorExe
                src/main.c
            )
target_link_libraries(TensorExe
                        PRIVATE
                            TensorLib
)
