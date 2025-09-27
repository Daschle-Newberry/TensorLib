#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H
#include "tensor.h"
typedef enum {
    TENSOR_OP_ERROR_NONE,
    TENSOR_OP_ERROR_INTERNAL,
    TENSOR_OP_CANNOT_BROADCAST,
    TENSOR_OP_ERROR_SHAPE_MISMATCH,
    TENSOR_OP_ERROR_COUNT,
}TensorOperationError;

TensorOperationError broadcast_shapes(Tensor* a, Tensor* b);

TensorOperationError add_tensor(Tensor* out, const Tensor* a, const Tensor* b);
TensorOperationError sub_tensor(Tensor* out, const Tensor* a, const Tensor* b);
TensorOperationError mul_tensor(Tensor* out, const Tensor* a, const Tensor* b);
TensorOperationError div_tensor(Tensor* out, const Tensor* a, const Tensor* b);

const char* tensor_operation_error_to_string(TensorOperationError err);
#endif //TENSOR_OPERATIONS_H