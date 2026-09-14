#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef enum {
  TENSOR_ERROR_NONE = 0,
  TENSOR_ERROR_NO_MEMORY = 1,
  TENSOR_ERROR_INVALID_ARGUMENT = 2,
  TENSOR_ERROR_ARGUMENT_OVERFLOW = 3,
} TensorError;

typedef enum {
  T_FLOAT32,
  T_FLOAT64,
  T_INT32,
  T_INT64
} TensorDType;

typedef struct TensorStorage TensorStorage;
typedef struct Tensor Tensor;

/**
 * @brief Allocate an empty tensor (uninitialized data)
 *
 * @param out Tensor pointer to allocate the new tensor at
 * @param type The type of the tensor
 * @param ndim Number of dimensions
 * @param shape Array of length ndim specifying the size of each dimension.
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_init_empty(
    Tensor* out, 
    TensorDType type, 
    size_t ndim, 
    const size_t shape[ndim]
    );



Tensor* tensor_create_empty(
    TensorDType type,
    size_t ndim,
    const size_t shape[ndim],
    TensorError* err
    );

/**
 * @brief Free ALL heap memory associated with the tensor (data, shape, strides).
 *
 * @param tensor Tensor to be freed
 */
void tensor_destroy(Tensor* tensor);



#endif
