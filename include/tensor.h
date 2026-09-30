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
 * @brief Creates an empty tensor (uninitialized data)
 *
 * @param type The type of the tensor
 * @param ndim Number of dimensions
 * @param shape Array of length ndim specifying the size of each dimension.
 * @param err Pointer to a TensorError for error codes
 *
 * @return Pointer to a heap allocated tensor on success, NULL otherwise
 */
Tensor* tensor_create_empty(
    TensorDType type,
    size_t ndim,
    const size_t shape[ndim],
    TensorError* err
    );

/**
 * @brief Creates a tensor and fills it with the given data
 *
 * @param type The type of the tensor
 * @param ndim Number of dimensions
 * @param shape Array of length ndim specifying the size of each dimension.
 * @param data Array of length product(shape) specifying the data to use
 * @param err Pointer to a TensorError for error codes
 *
 * @return Pointer to a heap allocated tensor on success, NULL otherwise
 */
Tensor* tensor_create_from_data(
    TensorDType type,
    size_t ndim,
    const size_t shape[ndim],
    const void* data,
    TensorError* err
    );

/**
 * @brief Free ALL heap memory associated with the tensor (data, shape, strides).
 *
 * @param tensor Tensor to be freed
 */
void tensor_destroy(Tensor* tensor);

/**
 * @brief Generates a string representation of the given tensor.
 *
 * @param tensor Tensor to be stringified
 *
 * @return Pointer to a heap allocated null terminated string
 */
char* tensor_to_string(const Tensor* tensor);

#endif
