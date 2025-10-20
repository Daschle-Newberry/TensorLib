#define MAX(a,b)((a) > (b) ? (a) : (b))
#define MIN(a,b)((a) < (b) ? (a) : (b))

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "tensor.h"


TensorError elementwise_broadcast(Tensor* out, const Tensor* a, const Tensor* b) {
    const int max_ndim = MAX(a->ndim, b->ndim);
    const int min_ndim = MIN(a->ndim, b->ndim);
    const int diff = max_ndim - min_ndim;


    int shape[max_ndim];

    for (int i = 0; i < diff; i++) {

    }


    return TENSOR_ERROR_NONE;
}

TensorError matrix_broadcast(Tensor* out, Tensor* a, Tensor* b) {
    if (a->ndim != b->ndim) return TENSOR_ERROR_CANNOT_BROADCAST;
    int out_shape[a->ndim];

    for (int i = 0; i < a->ndim - 2; i++) {
        if (a->shape[i] != b->shape[i] && a->shape[i] != 1 && b->shape[i] != 1){
            return TENSOR_ERROR_CANNOT_BROADCAST;
        }
        out_shape[i] = a->shape[i] != 1 ? a->shape[i] : b->shape[i];
    }

    out_shape[a->ndim - 2] = a->shape[a->ndim - 2];
    out_shape[a->ndim - 1] = b->shape[b->ndim - 1];

    const TensorError err = tensor_init(out,NULL,out_shape, a->ndim);

    if (err != TENSOR_ERROR_NONE) return err;

    out_shape[a->ndim - 1] = a->shape[a->ndim - 1];
    out_shape[a->ndim - 2] = a->shape[a->ndim - 2];
    tensor_broadcast_to(a,out->shape,out->ndim, true);

    out_shape[a->ndim - 1] = b->shape[b->ndim - 1];
    out_shape[a->ndim - 2] = b->shape[b->ndim - 2];
    tensor_broadcast_to(b, out->shape, out->ndim,true);

    return TENSOR_ERROR_NONE;
}

TensorError element_wise_operation(Tensor* out, const Tensor* a, const Tensor* b, float (*op)(float,float)) {
    Tensor a_copy;
    Tensor b_copy;

    const Tensor* max_tensor = a->ndim < b->ndim ? b : a;
    const Tensor* min_tensor = a->ndim < b->ndim ? a : b;

    tensor_shallow_copy(&a_copy,max_tensor,max_tensor->ndim);
    tensor_shallow_copy(&b_copy,min_tensor,max_tensor->ndim);

    const TensorError err = elementwise_broadcast(out, &a_copy,&b_copy);

    if (err != TENSOR_ERROR_NONE) return err;

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

    return TENSOR_ERROR_NONE;
}

TensorError tensor_mat_mul(Tensor* out, const Tensor* a, const Tensor* b) {
    Tensor a_copy;
    Tensor b_copy;

    const int total_dims = MAX(2,MAX(a->ndim,b->ndim));

    tensor_shallow_copy(&a_copy,a,total_dims);
    tensor_shallow_copy(&b_copy,b,total_dims);

    if (b->ndim == 1) {
        tensor_promote_to_col(&b_copy);
    }

    const int a_k = a_copy.shape[a_copy.ndim - 1];
    const int b_k = b_copy.shape[b_copy.ndim - 2];

    if (a_k != b_k) {return TENSOR_ERROR_INPUT_DIM_MISMATCH;}

    const TensorError err = matrix_broadcast(out, &a_copy, &b_copy);

    if (err != TENSOR_ERROR_NONE) return err;


}
float add_op(const float x, const float y) {return x + y;}
float sub_op(const float x, const float y) {return x - y;}
float mul_op(const float x, const float y) {return x * y;}
float div_op(const float x, const float y) {return x / y;}

TensorError tensor_add(Tensor* out, const Tensor* a, const Tensor* b) {return element_wise_operation(out,a,b,add_op);}
TensorError tensor_sub(Tensor* out, const Tensor* a, const Tensor* b) {return element_wise_operation(out,a,b,sub_op);}
TensorError tensor_mul(Tensor* out, const Tensor* a, const Tensor* b) {return element_wise_operation(out,a,b,mul_op);}
TensorError tensor_div(Tensor* out, const Tensor* a, const Tensor* b) {return element_wise_operation(out,a,b,div_op);}

