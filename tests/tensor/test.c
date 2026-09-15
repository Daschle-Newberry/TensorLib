#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include "harness.h"
#include "tensor.h"
#include "private/tensor_private.h"

TEST(tensor_create_empty) {
  TensorError err;
  Tensor* tensor;
  
  // ******************** CREATE 2 DIM F32 ******************** 
  tensor = tensor_create_empty(T_FLOAT32, 2, (size_t[]){3,3}, &err);
  ASSERT(err == TENSOR_ERROR_NONE);
  ASSERT(tensor != NULL);
  ASSERT(tensor->ndim = 2);
  ASSERT(tensor->storage != NULL);
  ASSERT(tensor->storage->n == 9 * sizeof(float));
  ASSERT_ARRAY_EQ(tensor->shape,((size_t[]) {3,3}), 2, size_t);
  ASSERT_ARRAY_EQ(tensor->strides, ((size_t[]){3 * sizeof(float),sizeof(float)}), 2, size_t);

  tensor_destroy(tensor);
 
  // ******************** CREATE 1 DIM F32 ******************** 
  tensor = tensor_create_empty(T_FLOAT32, 1, (size_t[]){3}, &err);
  ASSERT(err == TENSOR_ERROR_NONE);
  ASSERT(tensor != NULL);
  ASSERT(tensor->ndim = 1);
  ASSERT(tensor->storage != NULL);
  ASSERT(tensor->storage->n == 3 * sizeof(float));
  ASSERT_ARRAY_EQ(tensor->shape, ((size_t[]){3}), 1, size_t);
  ASSERT_ARRAY_EQ(tensor->strides, ((size_t[]){sizeof(float)}), 1, size_t);

  tensor_destroy(tensor);

  // ******************** CREATE 0 DIM F32 ******************** 
  tensor = tensor_create_empty(T_FLOAT32, 0, NULL, &err);
  ASSERT(err == TENSOR_ERROR_NONE);
  ASSERT(tensor != NULL);
  ASSERT(tensor->ndim == 0);
  ASSERT(tensor->storage != NULL);
  ASSERT(tensor->storage->n == sizeof(float));
  ASSERT(tensor->shape == NULL);
  ASSERT(tensor->strides == NULL);

  tensor_destroy(tensor);

  // Direct Overflow
  tensor = tensor_create_empty(T_FLOAT32, 2, (size_t[]){LONG_MAX, LONG_MAX},&err);
  ASSERT(tensor == NULL);
  ASSERT(err == TENSOR_ERROR_ARGUMENT_OVERFLOW);

  //Indirect Overflow 
  tensor = tensor_create_empty(T_FLOAT32, 2, (size_t[]){sqrt(LONG_MAX), sqrt(LONG_MAX)},&err);
  ASSERT(tensor == NULL);
  ASSERT(err == TENSOR_ERROR_ARGUMENT_OVERFLOW); 
}


TEST(tensor_destroy) {
  TensorError err;
  Tensor* tensor = tensor_create_empty(T_FLOAT32, 2, (size_t[]){3,3}, &err);
  
  TensorStorage* storage = tensor->storage;
  storage->refs++;
  
  ((int*)storage->data)[0] = 1;

  tensor_destroy(tensor);

  ASSERT(storage->refs == 1);
  ASSERT(((int*)storage->data)[0] == 1);

  free(storage->data);
  free(storage);
}

int main() {
  
  RUN(tensor_create_empty);
  RUN(tensor_destroy);

  return 0;
}

