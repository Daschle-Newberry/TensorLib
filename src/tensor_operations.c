#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "tensor.h"
#include "tensor_operations.h"


size_t one_dim_length(const size_t* shape, const size_t ndim) {
    size_t total = 0;
    for (size_t i = 0; i < ndim; i++) total *= shape[i];
    return total;
}

TensorOperationError broadcast_shapes(Tensor* a, Tensor* b) {
    const Tensor* max_tensor = a->ndim < b->ndim ? b : a;
    Tensor* min_tensor = a->ndim < b->ndim ? a : b;

    for (size_t i = 0; i < min_tensor->ndim; i++) {
        if (a->shape[a->ndim - 1 - i] != b->shape[b->ndim - 1- i]) {
            return TENSOR_OP_CANNOT_BROADCAST;
        }
    }
    size_t new_shape[max_tensor->ndim];

    size_t added_dims = max_tensor->ndim - min_tensor->ndim;
    for (size_t i = 0; i < added_dims; i++) {
        new_shape[i] = 1;
    }

    for (size_t i = 0; i < min_tensor->ndim; i++) {
        new_shape[added_dims + i] = min_tensor->shape[i];
    }

    tensor_reshape(min_tensor,min_tensor->data,new_shape,max_tensor->ndim);

    for (size_t i = 0; i < max_tensor->ndim; i++) {
        if (min_tensor->shape[i] == 1) {
            min_tensor->shape[i] = max_tensor->shape[i];
            min_tensor->strides[i] = 0;
        }else if (max_tensor->shape[i] == 1) {
            max_tensor->shape[i] = min_tensor->shape[i];
            max_tensor->strides[i] = 0;
        }
    }
    return TENSOR_OP_ERROR_NONE;

}

TensorOperationError elementWiseOperation(Tensor* out, const Tensor* a, const Tensor* b, float (*op)(float,float)) {
    Tensor a_copy;
    Tensor b_copy;

    tensor_shallow_copy(&a_copy,a);
    tensor_shallow_copy(&b_copy,b);

    broadcast_shapes(&a_copy,&b_copy);

    tensor_init(out,NULL,a_copy.shape,a_copy.ndim);

    size_t total = 1;
    for (size_t d = 0; d < out->ndim; d++) total *= out->shape[d];

    for (size_t idx = 0; idx < total; idx++) {
        size_t offset_a = 0;
        size_t offset_b = 0;
        size_t tmp = idx;
        for (size_t dim = 0; dim < out->ndim; dim++) {
            const size_t d_idx = tmp / out->strides[dim];
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

TensorOperationError add_tensor(const Tensor* out, const Tensor* a, const Tensor* b) {return elementWiseOperation(out,a,b,add_op);}
TensorOperationError sub_tensor(const Tensor* out, const Tensor* a, const Tensor* b) {return elementWiseOperation(out,a,b,sub_op);}
TensorOperationError mul_tensor(const Tensor* out, const Tensor* a, const Tensor* b) {return elementWiseOperation(out,a,b,mul_op);}
TensorOperationError div_tensor(const Tensor* out, const Tensor* a, const Tensor* b) {return elementWiseOperation(out,a,b,div_op);}
