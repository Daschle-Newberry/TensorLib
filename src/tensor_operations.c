#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "tensor.h"
#include "tensor_operations.h"

static const char* TensorOperationErrorStrings[] = {
    [TENSOR_OP_ERROR_NONE] = "TENSOR_OP_ERROR_NONE",
    [TENSOR_OP_ERROR_INTERNAL] = "TENSOR_OP_ERROR_INTERNAL",
    [TENSOR_OP_CANNOT_BROADCAST] = "TENSOR_OP_CANNOT_BROADCAST",
    [TENSOR_OP_ERROR_SHAPE_MISMATCH] = "TENSOR_OP_ERROR_SHAPE_MISMATCH",
    [TENSOR_OP_ERROR_COUNT] = "TENSOR_OP_ERROR_COUNT",
};

int one_dim_length(const int* shape, const int ndim) {
    int total = 0;
    for (int i = 0; i < ndim; i++) total *= shape[i];
    return total;
}

TensorOperationError broadcast(Tensor* out, Tensor* a, Tensor* b) {
    if (a->ndim != b->ndim) return TENSOR_OP_CANNOT_BROADCAST;
    int out_shape[a->ndim];

    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i] && a->shape[i] != 1 && b->shape[i] != 1){
            return TENSOR_OP_CANNOT_BROADCAST;
        }
        out_shape[i] = a->shape[i] != 1 ? a->shape[i] : b->shape[i];
    }

    const TensorError err = tensor_init(out,NULL,out_shape, a->ndim);

    if (err != TENSOR_ERROR_NONE) return TENSOR_OP_ERROR_INTERNAL;

    tensor_broadcast_to(a,out->shape,out->ndim);
    tensor_broadcast_to(b, out->shape, out->ndim);

    return TENSOR_OP_ERROR_NONE;
}

TensorOperationError elementWiseOperation(Tensor* out, const Tensor* a, const Tensor* b, float (*op)(float,float)) {
    Tensor a_copy;
    Tensor b_copy;

    const Tensor* max_tensor = a->ndim < b->ndim ? b : a;
    const Tensor* min_tensor = a->ndim < b->ndim ? a : b;

    tensor_shallow_copy(&a_copy,max_tensor,max_tensor->ndim);
    tensor_shallow_copy(&b_copy,min_tensor,max_tensor->ndim);

    const TensorOperationError op_err = broadcast(out, &a_copy,&b_copy);

    if (op_err != TENSOR_OP_ERROR_NONE) return op_err;

    int total = 1;
    for (int d = 0; d < out->ndim; d++) total *= out->shape[d];

    for (int idx = 0; idx < total; idx++) {
        int offset_a = 0;
        int offset_b = 0;
        int tmp = idx;
        for (int dim = 0; dim < out->ndim; dim++) {
            const int d_idx = tmp / out->strides[dim];
            tmp %= out->strides[dim];

            offset_a += d_idx * a_copy.strides[dim];
            offset_b += d_idx * b_copy.strides[dim];
        }

        out->data[idx] = op(a_copy.data[offset_a], b_copy.data[offset_b]);
    }

    return TENSOR_OP_ERROR_NONE;
}

float add_op(const float x, const float y) {return x + y;}
float sub_op(const float x, const float y) {return x - y;}
float mul_op(const float x, const float y) {return x * y;}
float div_op(const float x, const float y) {return x / y;}

TensorOperationError add_tensor(Tensor* out, const Tensor* a, const Tensor* b) {return elementWiseOperation(out,a,b,add_op);}
TensorOperationError sub_tensor(Tensor* out, const Tensor* a, const Tensor* b) {return elementWiseOperation(out,a,b,sub_op);}
TensorOperationError mul_tensor(Tensor* out, const Tensor* a, const Tensor* b) {return elementWiseOperation(out,a,b,mul_op);}
TensorOperationError div_tensor(Tensor* out, const Tensor* a, const Tensor* b) {return elementWiseOperation(out,a,b,div_op);}

const char* tensor_operation_error_to_string(const TensorOperationError err){return TensorOperationErrorStrings[err];}