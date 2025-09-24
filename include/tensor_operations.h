#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H
#include "tensor.h"
typedef enum {
    TENSOR_OP_ERROR_NONE,
    TENSOR_OP_CANNOT_BROADCAST,
    TENSOR_OP_ERROR_SHAPE_MISMATCH,
}TensorOperationError;

TensorOperationError broadcast_shapes(Tensor* a, Tensor* b);

TensorOperationError add_tensor(const Tensor* out, const Tensor* a, const Tensor* b);
TensorOperationError sub_tensor(const Tensor* out, const Tensor* a, const Tensor* b);
TensorOperationError mul_tensor(const Tensor* out, const Tensor* a, const Tensor* b);
TensorOperationError div_tensor(const Tensor* out, const Tensor* a, const Tensor* b);

#endif //TENSOR_OPERATIONS_H