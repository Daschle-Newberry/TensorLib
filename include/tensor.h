#ifndef TENSOR_H
#define TENSOR_H
#include <stdbool.h>
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

/**
 * Tensor structure representing an N-dimensional array of floats
 */
typedef struct {
    int ndim;     //< Number of dimensions
    int length;   //< Length of the contiguous data array
    int* shape;   //< Pointer to an array containing the sizes of each dimension
    int* strides; //< Pointer to an array containing the strides of each dimension
    float* data;  //< Pointer to an array of floats
} Tensor;

//TENSOR

/**
 * Allocate an empty tensor (uninitialized data)
 * @param out Tensor pointer to allocate the new tensor at
 * @param shape Array of length ndim specifying the size of each dimension.
 * @param ndim Number of dimensions
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_empty(Tensor* out, const int* shape, int ndim);

/**
 * Allocate and new tensor with a copy of the given data
 * @param out Tensor pointer to allocate the new tensor at
 * @param data Array of floats
 * @param shape Array of length ndim specifying the size of each dimension
 * @param ndim Number of dimensions
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_from_data(Tensor* out, const float* data, const int* shape,int ndim);

/**
 * Allocate a new tensor filled with zeros
 * @param out Tensor pointer to allocate the new tensor at
 * @param shape Array of length ndim specifying the size of each dimension
 * @param ndim Number of dimensions
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_zeros(Tensor* out,const int* shape,int ndim);

/**
 * Allocate a new tensor filled with ones
 * @param out Tensor pointer to allocate the new tensor at
 * @param shape Array of length ndim specifying the size of each dimension
 * @param ndim Number of dimensions
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_ones(Tensor* out,const int* shape,int ndim);

/**
 * Allocate a new tensor filled with a given value
 * @param out Tensor pointer to allocate the new tensor at
 * @param num The float the tensor will be filled with
 * @param shape Array of length ndim specifying the size of each dimension
 * @param ndim Number of dimensions
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_fill(Tensor* out, float num, const int* shape,int ndim);

/**
 * Free ALL memory associated with the tensor (data, shape, strides)
 * After calling, the tensor should NOT be used again
 * @param tensor Tensor to be freed
 */
void tensor_free(Tensor* tensor);

/**
 * Free just the tensor's metadata (shape,strides) and not the underlying shared data buffer.
 * After calling, the tensor should NOT be used again
 * @param tensor Tensor to be freed
 */
void tensor_view_free(Tensor* tensor);

/**
 *
 * @param tensor Tensor to read from
 * @param idx Integer array representing multidimensional indices
 * @return the value at those indices
 */
float tensor_get(const Tensor* tensor, const int* idx);

/**
 * Takes the given tensor and expands the dimensions to fit the new shape
 * @param out Tensor pointer to allocate the new tensor at
 * @param in Original tensor pointer
 * @param new_shape Array of length new_ndim specifying the new size of each dimension
 * @param new_ndim  New number of dimensions
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_expand(Tensor* out, const Tensor* in,
                          const int* new_shape, int new_ndim);

/**
 * Takes a one dimensional tensor and promotes it to a 2D column vector (Shape: [N,1])
 * @param out Tensor pointer to allocate the new tensor at
 * @param in Original tensor pointer
 * @return TENSOR_ERROR_NONE on success, error code otherwise
 */
TensorError tensor_promote_to_col(Tensor* out, const Tensor* in);


/**
 * Creates a string representing the tensor's data and shape
 * @param tensor Tensor to create a string from
 * @return String representing the tensor
 */
const char* tensor_to_string(const Tensor* tensor);

/**
 * Creates a string representing the tensor's metadata (shape, strides, ndim)
 * @param tensor Tensor to create a string from
 * @return String representing the tensor's metadata
 */
const char* tensor_metadata_to_string(const Tensor* tensor);

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

