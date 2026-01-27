#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
/**
 * Tensor error enum for different initialization or operation errors
 */


typedef enum {
    TENSOR_ERROR_NONE,
    TENSOR_ERROR_NO_MEMORY,
    TENSOR_ERROR_INVALID_ARGUMENT,
    TENSOR_ERROR_INPUT_DIM_MISMATCH,
    TENSOR_ERROR_NEGATIVE_DIM,
    TENSOR_ERROR_CANNOT_BROADCAST,
    TENSOR_ERROR_CANNOT_EXPAND,
    TENSOR_ERROR_COUNT,
}TensorError;

typedef enum{
    T_TYPE_INT,
    T_TYPE_FLOAT,
    T_TYPE_DOUBLE,
    T_TYPE_LONG,
}TensorType;

/**
 * Tensor structure representing an N-dimensional array of floats
 */
// typedef struct {
//     int ndim;     //< Number of dimensions
//     int length;   //< Length of the contiguous data array
//     int* shape;   //< Pointer to an array containing the sizes of each dimension
//     int* strides; //< Pointer to an array containing the strides of each dimension
//     float* data;  //< Pointer to an array of floats
// } Tensor;

typedef struct {
    size_t ndim;
    size_t length;
    size_t* shape;
    size_t* strides;
    uint8_t* data;
    size_t dbytes;
    TensorType dtype;
} Tensor;

//TENSOR

/**
 * Allocate an empty tensor (uninitialized data)
 * @param out Tensor pointer to allocate the new tensor at
 * @param shape Array of length ndim specifying the size of each dimension.
 * @param ndim Number of dimensions
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_init_empty(Tensor* out, const size_t* shape, size_t ndim, TensorType dtype);

/**
 * Allocate and new tensor with a copy of the given data
 * @param out Tensor pointer to allocate the new tensor at
 * @param data Array of floats
 * @param shape Array of length ndim specifying the size of each dimension
 * @param ndim Number of dimensions
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_init_from_data(Tensor* out, const void* data, const size_t* shape, size_t ndim, TensorType dtype);

/**
 * Allocate a new tensor filled with zeros
 * @param out1 Tensor pointer to allocate the new tensor at
 * @param shape Array of length ndim specifying the size of each dimension
 * @param ndim Number of dimensions
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_init_zeros(Tensor* out, const size_t* shape, size_t ndim, TensorType dtype);

/**
 * Allocate a new tensor filled with ones
 * @param out Tensor pointer to allocate the new tensor at
 * @param shape Array of length ndim specifying the size of each dimension
 * @param ndim Number of dimensions
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_init_ones(Tensor* out, const size_t* shape, size_t ndim, TensorType dtype);

/**
 * Allocate a new tensor filled with a given value
 * @param out Tensor pointer to allocate the new tensor at
 * @param num The float the tensor will be filled with
 * @param shape Array of length ndim specifying the size of each dimension
 * @param ndim Number of dimensions
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_init_fill(Tensor* out, const void* num, const size_t* shape,size_t ndim, TensorType dtype);

/**
 * Free ALL heap memory associated with the tensor (data, shape, strides).
 * After calling, the tensor should NOT be used again
 * @param tensor Tensor to be freed
 */
void tensor_destroy(Tensor* tensor);

/**
 * Free just the tensor's metadata (shape,strides) and not the underlying shared data buffer.
 * After calling, the tensor should NOT be used again
 * @param tensor Tensor to be freed
 */
void tensor_view_destroy(Tensor* tensor);

/**
 *  * Takes the given tensor and broadcasts the dimensions to fit the new shape
 * @param out Tensor pointer to allocate the new tensor at
 * @param in Original tensor pointer
 * @param new_shape Array of length new_ndim specifying the new size of each dimension
 * @param new_ndim  New number of dimensions
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_broadcast_to(Tensor* out, const Tensor* in,
                          const size_t* new_shape, size_t new_ndim);

/**
 * Takes the given tensor and broadcasts all dimensions except the last two to fit the new shape
 * @param out Tensor pointer to allocate the new tensor at
 * @param in Original tensor pointer
 * @param new_shape Array of length new_ndim specifying the new size of each dimension
 * @param new_ndim  New number of dimensions
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_matrix_broadcast_to(Tensor* out, const Tensor* in,
                          const size_t* new_shape, size_t new_ndim);

/**
 * Takes a one dimensional tensor and promotes it to a 2D column vector (Shape: [N,1])
 * @param out Tensor pointer to allocate the new tensor at
 * @param in Original tensor pointer
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_promote_to_col(Tensor* out, const Tensor* in);

/**
 * Takes a one dimensional tensor and promotes it to a 2D row vector (Shape: [1,N])
 * @param out Tensor pointer to allocate the new tensor at
 * @param in Original tensor pointer
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_promote_to_row(Tensor* out, const Tensor* in);

/**
 * Creates a string representing the tensor's data and shape
 * @param tensor Tensor to create a string from
 * @return Heap allocated string representing the tensor, called must free
 */
char* tensor_to_string(const Tensor* tensor);

/**
 * Creates a string representing the tensor's metadata (shape, strides, ndim)
 * @param tensor Tensor to create a string from
 * @return Heap allocated string representing the tensor's metadata, called must free
 */
char* tensor_metadata_to_string(const Tensor* tensor);

/**
 * Creates a string from the TensorError enum
 * @param error Error code
 * @return String representing the error code
 */
const char* tensor_error_to_string(TensorError error);

// TENSOR_OP

/**
 * Matrix multiplication between two tensors, broadcasting if possible
 * Uses the last two dimensions as the matrix dimensions, all other dimensions are treated as batches
 *
 * @param out Tensor pointer to allocate the resulting tensor at
 * @param a Left tensor
 * @param b Right tensor
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_mat_mul(Tensor* out, const Tensor* a, const Tensor* b);

/**
 * Elementwise addition between two tensors, broadcasting if possible
 *
 * @param out Tensor pointer to allocate the resulting tensor at
 * @param a Left tensor
 * @param b Right tensor
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_add(Tensor* out, const Tensor* a, const Tensor* b);

/**
 * Elementwise subtraction between two tensors, broadcasting if possible
 *
 * @param out Tensor pointer to allocate the resulting tensor at
 * @param a Left tensor
 * @param b Right tensor
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_sub(Tensor* out, const Tensor* a, const Tensor* b);

/**
 * Elementwise multiplication between two tensors, broadcasting if possible
 *
 * @param out Tensor pointer to allocate the resulting tensor at
 * @param a Left tensor
 * @param b Right tensor
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_mul(Tensor* out, const Tensor* a, const Tensor* b);

/**
 * Elementwise division between two tensors, broadcasting if possible
 *
 * @param out Tensor pointer to allocate the resulting tensor at
 * @param a Left tensor
 * @param b Right tensor
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_div(Tensor* out, const Tensor* a, const Tensor* b);

#endif //TENSOR_H

