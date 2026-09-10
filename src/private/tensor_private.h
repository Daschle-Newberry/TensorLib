#ifndef TENSOR_PRIVATE_H
#define TENSOR_PRIVATE_H

#include <stddef.h>

#include "tensor.h"

struct TensorStorage {
  size_t refs;          //< Number of references to this storage
  size_t n;             //< Length of the data array in bytes
  void* data;           //< Data array, length n
};

struct Tensor {
  TensorStorage* storage;   //< Pointer to data storage
  TensorDType type;         //< Type of the tensor
  size_t ndim;              //< Number of dimensions in the tensor
  size_t* shape;            //< Array representing the tensor shape, length ndim
  size_t* strides;          //< Array of strides for each dim, length ndim
};

#endif

