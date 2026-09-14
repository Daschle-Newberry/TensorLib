#include <stdlib.h>

#include "harness.h"
#include "tensor.h"
#include "private/tensor_private.h"

TEST(tensor_init_empty) {
  TensorError err;
  Tensor* tensor = tensor_create_empty(T_FLOAT32, 2, (size_t[]){3,3}, &err);
  ASSERT(err == TENSOR_ERROR_NONE);
  ASSERT(tensor != NULL);
  
  ASSERT(0);
  ASSERT(1 == 0);

  free(tensor->shape);
  free(tensor->strides);
  free(tensor->storage->data);
  free(tensor->storage);
  free(tensor);
}

int main() {
  
  RUN(tensor_init_empty);

  return 0;
}

