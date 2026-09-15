#include <stdint.h>
#include <stdio.h>
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
  }
  return 0;
}

static inline TensorError contiguous_size(
    size_t dbytes,
    size_t ndim, 
    const size_t shape[ndim],
    size_t* out_bytes
) {
  // Scalar
  if(ndim == 0) {
    *out_bytes = dbytes;
    return TENSOR_ERROR_NONE;
  }

  size_t len = 1;

  bool empty = false;
  for(size_t i = 0; i < ndim; i++) {
    size_t dim = shape[i];
    
    if(dim == 0) {
      empty = true;
      dim = 1;
    }

    if(len > SIZE_MAX / dim)
      return TENSOR_ERROR_ARGUMENT_OVERFLOW;

    len *= shape[i]; 
  }
  
  if(len > SIZE_MAX / dbytes)
    return TENSOR_ERROR_ARGUMENT_OVERFLOW;

  *out_bytes = empty ? 0 : len * dbytes;

  return TENSOR_ERROR_NONE;
}

// Precondition: continguous_size has succeeded for the same dbytes and shape
static inline void compute_strides(
    size_t dbytes,
    size_t ndim, 
    const size_t shape[ndim], 
    size_t strides[ndim]
) {
  strides[ndim - 1] = dbytes;
  for(size_t i = ndim - 1; i > 0; i--) 
    strides[i - 1] = strides[i] * shape[i];
}

Tensor* tensor_create_empty(
    TensorDType type,
    size_t ndim,
    const size_t shape[ndim],
    TensorError* err
) {

  Tensor* tensor         = NULL;
  TensorStorage* storage = NULL;
  void* buff             = NULL;
  
  TensorError error = TENSOR_ERROR_NONE;
  
  size_t dbytes = dtype_size(type);
  if((ndim == 0) != (shape == NULL) || dbytes == 0) {
    error = TENSOR_ERROR_INVALID_ARGUMENT;
    goto cleanup;
  }
 
  size_t buff_size;
  error = contiguous_size(dbytes, ndim, shape, &buff_size);

  if(error)
    goto cleanup;

  // Allocate extra space for metadata
  tensor = malloc(sizeof(*tensor) + 2 * ndim * sizeof(*tensor->meta));
  storage = malloc(sizeof(*storage));
  buff = malloc(buff_size);

  if(!tensor || !storage || !buff) {
    error = TENSOR_ERROR_NO_MEMORY;
    goto cleanup;
  }

  *storage = (TensorStorage)
  {
    .refs = 1,
    .n = buff_size,
    .data = buff
  };

  *tensor = (Tensor) 
  {
    .storage = storage,
    .type    = type,
    .ndim    = ndim,
    .shape   = NULL,
    .strides = NULL,
  };
  
  if(ndim > 0) {
    tensor->shape = tensor->meta;
    tensor->strides = tensor->meta + ndim;
    memcpy(tensor->shape, shape, ndim * sizeof(*shape));
    compute_strides(dbytes, ndim, tensor->shape, tensor->strides);
  }

  if(err)
    *err = error;
  return tensor;

cleanup:
  free(tensor);
  free(storage);
  free(buff);
  if(err)
    *err = error;
  return NULL;
}

void tensor_destroy(Tensor* tensor) {
  if(!tensor)
    return;
  
  if(tensor->storage->refs == 1) {
    free(tensor->storage->data);
    free(tensor->storage);
  } else {
    tensor->storage->refs--;
  }
  free(tensor);
}
