# TensorLib

## About

---
TensorLib is a C library that is capable of doing linear algebra 
operations on N-dimension tensors, without memory copies. It does this by using strides
and broadcasting, resulting in shared data but different tensor interpretations. TensorLib is heavily inspired
by other libraries, and aims to achieve similar results to NumPy.


## How to Build (Library)

---

### Prerequisites

- **C compiler (e.g., GCC or Clang)** 
- **CMake ≥ 3.10 (https://cmake.org/download/)**

### Build steps

1. Clone the repo by using the following command in the terminal
```commandline
git clone https://github.com/Daschle-Newberry/TensorLib
cd TensorLib
```
2. Generate build files

```commandline
cmake -B build -G "Your Build System"
```

3. Build the project
```commandline
cmake --build build
```

## Example
```c++
#inlcude "tensor.h"

void main(){
    Tensor out;
    Tensor a;
    Tensor b;
    
    //Intialize tensor a
    tensor_from_data(&a, (float[]){1,2,3,4}, (int[]){2,2},2);
    
    //Intialize tensor b
    tensor_from_data(&b, (float[]){1,2}, (int[]){1,2},2);
    
    //Compute the matrix multiplication of a and b
    tensor_mat_mul(&out, &a, &b);
    
    //Print the result
    printf("%s\n", tensor_to_string(&out)) 
}

```

## Features

---

### Tensor Handling

- Variety of initialization tools, including from data, empty, zeros, ones, or fill.
- Tensor view tools such as column promotion, expand, ect. 
- Debug and visualization tools such as metadata to string or tensor to string.

### Tensor Operations
- Elementwise broadcasting
- Matrix broadcasting
- Elementwise addition, subtraction, multiplication, and division.
- ~~Matrix multiplication~~
- ~~Matrix transpose~~
- ~~Scalar multiplication~~

### Optimizations
- ~~SIMD~~
- ~~GPU acceleration~~
- ~~BLAS~~

