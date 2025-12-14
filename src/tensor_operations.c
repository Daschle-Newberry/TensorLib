#define MAX(a,b)((a) > (b) ? (a) : (b))
#define MIN(a,b)((a) < (b) ? (a) : (b))
#define MAX_TENSOR(a,b)(((a->ndim) >= (b->ndim)) ? (a) : (b))
#define MIN_TENSOR(a,b)(((a->ndim) < (b->ndim)) ? (a) : (b))

#include "tensor.h"

static int element_wise_broadcast(Tensor* out, const Tensor* a, const Tensor* b) {
    const Tensor* max = MAX_TENSOR(a,b);
    const Tensor* min = MIN_TENSOR(a,b);

    const int diff = max->ndim - min->ndim;

    int shape[max->ndim];

    for (int i = 0; i < diff; i++) {
        shape[i] = max->shape[i];
    }

    for (int i = diff; i < max->ndim; i++) {
        const int a_dim = max->shape[i];
        const int b_dim = min->shape[i - diff];

        if (a_dim == b_dim || b_dim == 1) shape[i] = a_dim;
        else if (a_dim == 1) shape[i] = b_dim;
        else return -1;
    }

    tensor_empty(out,(const int*) &shape,max->ndim);

    return 0;
}


static int matrix_broadcast(Tensor* out, const Tensor* a, const Tensor* b) {
    const Tensor* max = MAX_TENSOR(a,b);
    const Tensor* min = MIN_TENSOR(a,b);

    const int diff = max->ndim - min->ndim;

    int shape[max->ndim];

    for (int i = 0; i < diff; i++) {
        shape[i] = max->shape[i];
    }

    for (int i = diff; i < max->ndim - 2; i++) {
        const int a_dim = max->shape[i];
        const int b_dim = min->shape[i - diff];

        if (a_dim == b_dim || b_dim == 1) shape[i] = a_dim;
        else if (a_dim == 1) shape[i] = b_dim;
        else return -1;
    }

    shape[max->ndim - 1] = b->shape[b->ndim - 1];
    shape[max->ndim - 2] = a->shape[a->ndim - 2];

    tensor_empty(out,(const int*) &shape,max->ndim);

    return 0;
}


static TensorError element_wise_operation(Tensor* out, const Tensor* a, const Tensor* b, float (*op)(float,float)){
        if (element_wise_broadcast(out,a,b) < 0) return TENSOR_ERROR_CANNOT_BROADCAST;
        Tensor a_view, b_view;
        tensor_broadcast_to(&a_view, a, out->shape, out->ndim);
        tensor_broadcast_to(&b_view, b, out->shape, out->ndim);

        int total = 1;
        for (int d = 0; d < out->ndim; d++) total *= out->shape[d];

        for (int idx = 0; idx < total; idx++) {
            int offset_a = 0;
            int offset_b = 0;
            int tmp = idx;

            for (int dim = 0; dim < out->ndim; dim++) {
                const int d_idx = tmp / out->strides[dim];
                tmp %= out->strides[dim];

                offset_a += d_idx * a_view.strides[dim];
                offset_b += d_idx * b_view.strides[dim];
            }

            out->data[idx] = op(a_view.data[offset_a], b_view.data[offset_b]);
        }

        return TENSOR_ERROR_NONE;
    }

TensorError tensor_mat_mul(Tensor* out, const Tensor* a, const Tensor* b) {
    Tensor a_tmp = *a;
    Tensor b_tmp = *b;

    if (a->ndim < 2) tensor_promote_to_row(&a_tmp, a);
    if (b->ndim < 2) tensor_promote_to_col(&b_tmp, b);

    const int a_cols = a->shape[a->ndim - 1];
    const int b_rows = b->shape[b->ndim - 2];

    if (a_cols != b_rows) return TENSOR_ERROR_INPUT_DIM_MISMATCH;

    if (matrix_broadcast(out, &a_tmp,&b_tmp) < 0) return TENSOR_ERROR_CANNOT_BROADCAST;

    //NxM
    int shape[out->ndim];
    int total = out->length;
     for (int idx = 0; idx < total; idx++) {
         int tmp = idx;
         for (int dim = 0; dim < out->ndim; dim++) {
             shape[dim] = tmp % out->shape[dim];
             tmp /= out->shape[dim];
         }
     }
    return TENSOR_ERROR_NONE;
}
float add_op(const float x, const float y) {return x + y;}
float sub_op(const float x, const float y) {return x - y;}
float mul_op(const float x, const float y) {return x * y;}
float div_op(const float x, const float y) {return x / y;}

TensorError tensor_add(Tensor* out, const Tensor* a, const Tensor* b) {return element_wise_operation(out,a,b,add_op);}
TensorError tensor_sub(Tensor* out, const Tensor* a, const Tensor* b) {return element_wise_operation(out,a,b,sub_op);}
TensorError tensor_mul(Tensor* out, const Tensor* a, const Tensor* b) {return element_wise_operation(out,a,b,mul_op);}
TensorError tensor_div(Tensor* out, const Tensor* a, const Tensor* b) {return element_wise_operation(out,a,b,div_op);}

