#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "tensor.h"
#include "private/tensor_private.h"

static inline size_t dtype_size(TensorDType type) {
  switch(type) {
    case T_FLOAT32: return sizeof(float);
    case T_FLOAT64: return sizeof(double);
    case T_INT32:   return sizeof(int32_t);
    case T_INT64:   return sizeof(int64_t);
    default:        return 0;
  }
}

static inline bool validate_shape(size_t ndim, const size_t* shape) {
  for(size_t i = 0; i < ndim; i++) {
    if(shape[i] == 0) 
      return false;
  }

  return true;
}

static inline TensorError get_data_buffer_size(
    size_t dbytes,
    size_t ndim, 
    const size_t shape[ndim],
    size_t* res
) {
  if(ndim == 0) {
    *res = dbytes;
    return TENSOR_ERROR_NONE;
  }

  size_t len = 1;

  for(size_t i = 0; i < ndim; i++) {
    if(len > SIZE_MAX / shape[i])
      return TENSOR_ERROR_ARGUMENT_OVERFLOW;

    len *= shape[i]; 
  }
  
  if(len > SIZE_MAX / dbytes)
    return TENSOR_ERROR_ARGUMENT_OVERFLOW;

  *res = len * dbytes;

  return TENSOR_ERROR_NONE;
}

// Overflow is assumed to be caught already by running get_data_buffer_size
static inline TensorError compute_strides(
    size_t dbytes,
    size_t ndim, 
    const size_t shape[ndim], 
    size_t strides[ndim]
) {
  strides[ndim - 1] = dbytes;
  for(size_t i = ndim - 1; i > 0; i--) { 
    strides[i - 1] = strides[i] * shape[i];
  }

  return TENSOR_ERROR_NONE;
}

TensorError tensor_init_empty(
    Tensor* out, 
    TensorDType type, 
    size_t ndim, 
    const size_t shape[ndim]
) {
  if(!out || (ndim != 0 && !shape) || !validate_shape(ndim, shape))
    return TENSOR_ERROR_INVALID_ARGUMENT;
  
  TensorError err = TENSOR_ERROR_NONE;

  size_t dbytes = dtype_size(type);

  if(dbytes == 0)
    return TENSOR_ERROR_INVALID_ARGUMENT;

  size_t buff_size;
  err = get_data_buffer_size(dbytes, ndim, shape, &buff_size);
  
  if(err)
    return err;

  // Predeclare so variables are in scope for cleanup
  TensorStorage* storage = NULL;
  void* buff = NULL;
  size_t* shape_copy = NULL;
  size_t* strides = NULL;

  storage = malloc(sizeof(*storage)); 
  buff = malloc(buff_size);

  if(!storage || !buff) {
    err = TENSOR_ERROR_NO_MEMORY;
    goto cleanup;
  }

  if(ndim > 0) {
    shape_copy = malloc(ndim * sizeof(*shape_copy));
    strides = malloc(ndim * sizeof(*strides));
  
    if(!shape_copy || !strides) {
      err = TENSOR_ERROR_NO_MEMORY;
      goto cleanup;
    }
    
    memcpy(shape_copy, shape, ndim * sizeof(*shape));
    err = compute_strides(dbytes, ndim, shape, strides);
  
    if(err) 
      goto cleanup;
  }

  *out = (Tensor) {
    .storage = storage,
    .type = type,
    .ndim = ndim,
    .shape = shape_copy,
    .strides = strides,
  };
  
  *out->storage = (TensorStorage) {
    .refs = 1,
    .n = buff_size,
    .data = buff,
  };

  return TENSOR_ERROR_NONE;


cleanup:
  free(storage);
  free(buff);
  free(shape_copy);
  free(strides);
  return err;
}

void tensor_destroy(Tensor* tensor) {

}
