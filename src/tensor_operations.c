#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "tensor.h"
#include "tensor_operations.h"


int one_dim_length(const int* shape, const int ndim) {
    int total = 0;
    for (int i = 0; i < ndim; i++) total *= shape[i];
    return total;
}

TensorOperationError broadcast(Tensor* out, Tensor* a, Tensor* b) {
    Tensor* max_tensor = a->ndim < b->ndim ? b : a;
    Tensor* min_tensor = a->ndim < b->ndim ? a : b;

    expand_dims(min_tensor, max_tensor->ndim);

    printf("%s",tensor_metadata_to_string(min_tensor));
    int out_shape[max_tensor->ndim];

    for (int i = 0; i < max_tensor->ndim; i++) {
        int s1 = max_tensor->shape[i];
        int s2 = min_tensor->shape[i];
        if (max_tensor->shape[i] != min_tensor->shape[i] && !(max_tensor->shape[i] == 1 || min_tensor->shape[i] == 1)){
            return TENSOR_OP_CANNOT_BROADCAST;
        }
        out_shape[i] = max_tensor->shape[i] != 1 ? max_tensor->shape[i] : min_tensor->shape[i];
    }

    TensorError err = tensor_init(out,NULL,out_shape, max_tensor->ndim);
    printf("%s\n",error_to_string(err));

    broadcast_to(max_tensor,out->shape,out->ndim);
    broadcast_to(min_tensor, out->shape, out->ndim);

    return TENSOR_OP_ERROR_NONE;
}
// TensorOperationError broadcast_shapes(Tensor* a, Tensor* b) {
//     const Tensor* max_tensor = a->ndim < b->ndim ? b : a;
//     Tensor* min_tensor = a->ndim < b->ndim ? a : b;
//
//     for (int i = 0; i < min_tensor->ndim; i++) {
//         if (a->shape[a->ndim - 1 - i] != b->shape[b->ndim - 1- i]) {
//             return TENSOR_OP_CANNOT_BROADCAST;
//         }
//     }
//     int new_shape[max_tensor->ndim];
//
//     int added_dims = max_tensor->ndim - min_tensor->ndim;
//     for (int i = 0; i < added_dims; i++) {
//         new_shape[i] = 1;
//     }
//
//     for (int i = 0; i < min_tensor->ndim; i++) {
//         new_shape[added_dims + i] = min_tensor->shape[i];
//     }
//
//     tensor_reshape(min_tensor,min_tensor->data,new_shape,max_tensor->ndim);
//
//     for (size_t i = 0; i < max_tensor->ndim; i++) {
//         if (min_tensor->shape[i] == 1) {
//             min_tensor->shape[i] = max_tensor->shape[i];
//             min_tensor->strides[i] = 0;
//         }else if (max_tensor->shape[i] == 1) {
//             max_tensor->shape[i] = min_tensor->shape[i];
//             max_tensor->strides[i] = 0;
//         }
//     }
//     return TENSOR_OP_ERROR_NONE;
//
// }

TensorOperationError elementWiseOperation(Tensor* out, const Tensor* a, const Tensor* b, float (*op)(float,float)) {
    Tensor a_copy;
    Tensor b_copy;

    tensor_shallow_copy(&a_copy,a);
    tensor_shallow_copy(&b_copy,b);

    const TensorOperationError op_err = broadcast(out, &a_copy,&b_copy);

    if (op_err != TENSOR_OP_ERROR_NONE) return op_err;

    // printf("A broadcasted %s", tensor_metadata_to_string(&a_copy));
    // printf("B broadcasted %s", tensor_metadata_to_string(&b_copy));
    // printf("OUT %s", tensor_metadata_to_string(out));


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
