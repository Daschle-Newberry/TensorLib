#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include "harness.h"
#include "tensor.h"
#include "private/tensor_private.h"

TEST(tensor_create_empty) {
  TensorError err;
  Tensor* tensor = tensor_create_empty(T_FLOAT32, 2, (size_t[]){3,3}, &err);
  ASSERT(err == TENSOR_ERROR_NONE);
  ASSERT(tensor != NULL);
  ASSERT(tensor->ndim = 2);
  ASSERT(tensor->storage != NULL);
  ASSERT(tensor->storage->n == 9 * sizeof(float));

  free(tensor->shape);
  free(tensor->strides);
  free(tensor->storage->data);
  free(tensor->storage);
  free(tensor);

  // Direct Overflow
  tensor = tensor_create_empty(T_FLOAT32, 2, (size_t[]){LONG_MAX, LONG_MAX},&err);
  ASSERT(tensor == NULL);
  ASSERT(err == TENSOR_ERROR_ARGUMENT_OVERFLOW);

  //Indirect Overflow 
  tensor = tensor_create_empty(T_FLOAT32, 2, (size_t[]){sqrt(LONG_MAX), sqrt(LONG_MAX)},&err);
  ASSERT(tensor == NULL);
  ASSERT(err == TENSOR_ERROR_ARGUMENT_OVERFLOW);
  
  // Invalid Shape
  tensor = tensor_create_empty(T_FLOAT32, 2, (size_t[]){0,1}, &err);
  ASSERT(tensor == NULL);
  ASSERT(err == TENSOR_ERROR_INVALID_ARGUMENT);

  // Invalid ndim 
  tensor = tensor_create_empty(T_FLOAT32, 0, (size_t[]){1,1}, &err);
  ASSERT(tensor == NULL);
  ASSERT(err == TENSOR_ERROR_INVALID_ARGUMENT);

}

int main() {
  
  RUN(tensor_create_empty);

  return 0;
}

