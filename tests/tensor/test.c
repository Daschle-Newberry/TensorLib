#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>

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

TEST(tensor_create_from_data) {
  TensorError err;
  Tensor* tensor;
  
  // ******************** CREATE 3 DIM F32 ******************** 
  float data3d[27];
  for(int i = 0; i < 27; i++) data3d[i] = i;
  
  tensor = tensor_create_from_data(T_FLOAT32, 3,(size_t[]){3, 3, 3}, &data3d, &err);
  ASSERT(err == TENSOR_ERROR_NONE);
  ASSERT(tensor != NULL);
  ASSERT_ARRAY_EQ(data3d, tensor->storage->data, 9, float);
 
  tensor_destroy(tensor);
 
  // ******************** CREATE 2 DIM F32 ******************** 
  float data2d[4];
  for(int i = 0; i < 4; i++) data3d[i] = i;
  
  tensor = tensor_create_from_data(T_FLOAT32, 2,(size_t[]){2, 2}, &data2d, &err);
  ASSERT(err == TENSOR_ERROR_NONE);
  ASSERT(tensor != NULL);
  ASSERT_ARRAY_EQ(data2d, tensor->storage->data, 4, float);
 
  tensor_destroy(tensor);
 
  // ******************** CREATE 1 DIM F32 ******************** 
  float data1d[2] = {1, 2};
  
  tensor = tensor_create_from_data(T_FLOAT32, 1,(size_t[]){2}, &data1d, &err);
  ASSERT(err == TENSOR_ERROR_NONE);
  ASSERT(tensor != NULL);
  ASSERT_ARRAY_EQ(data1d, tensor->storage->data, 2, float);
  
  tensor_destroy(tensor);
  
  // ******************** CREATE 1 DIM F32 ******************** 
  float data0d[1] = {1};
  
  tensor = tensor_create_from_data(T_FLOAT32, 0, NULL, &data0d, &err);
  ASSERT(err == TENSOR_ERROR_NONE);
  ASSERT(tensor != NULL);
  ASSERT_ARRAY_EQ(data0d, tensor->storage->data, 1, float);

  tensor_destroy(tensor);
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

TEST(tensor_to_string) {
  TensorError err;
  Tensor*     tensor;
  char*       str;
  // *********************** 1 F32 *********************** 
  tensor = tensor_create_empty(T_FLOAT32, 1, (size_t[]){1}, &err);
  str = tensor_to_string(tensor); 
  free(str);
  tensor_destroy(tensor);

  // *********************** 2x2 F32 *********************** 
  tensor = tensor_create_empty(T_FLOAT32, 2, (size_t[]){2,2}, &err);
  str = tensor_to_string(tensor); 
  free(str);
  tensor_destroy(tensor);

  // *********************** 3x3x3 F32 *********************** 
  tensor = tensor_create_empty(T_FLOAT32, 3, (size_t[]){3,3,3}, &err);
  str = tensor_to_string(tensor); 
  free(str);
  tensor_destroy(tensor);
}

int main() { 
  RUN(tensor_create_empty);
  RUN(tensor_create_from_data);
  RUN(tensor_destroy);
  RUN(tensor_to_string);
  return 0;
}

