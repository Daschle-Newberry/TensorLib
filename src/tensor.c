#include <stdlib.h>
#include <string.h>

#include "tensor.h"

static const char* TensorErrorStrings[] = {
    [TENSOR_ERROR_NONE] = "TENSOR_ERROR_NONE",
    [TENSOR_ERROR_NO_MEMORY] = "TENSOR_ERROR_NO_MEMORY",
    [TENSOR_ERROR_INPUT_DIM_MISMATCH] = "TENSOR_ERROR_INPUT_DIM_MISMATCH",
    [TENSOR_ERROR_NEGATIVE_DIM] = "TENSOR_ERROR_NEGATIVE_DIM",
    [TENSOR_ERROR_CANNOT_BROADCAST] = "TENSOR_ERROR_CANNOT_BROADCAST",
};

static int tensor_flat_length(const int* shape, int ndim) {
    int flat_length = 1;
    for (int i = 0; i < ndim; i++) {
        flat_length *= shape[i];
    }

    return flat_length;
}

static int tensor_alloc_data(Tensor* out, const int flat_length) {
    out->data = malloc(flat_length * sizeof *out->data);
    return out->data ? 0 : -1;
}

static int tensor_alloc_metadata(Tensor* out, const int* shape, const int ndim) {
    char* metadata = malloc(
       (ndim * sizeof *out->shape) + (ndim * sizeof *out->strides)
    );

    if (metadata == NULL) {
        return TENSOR_ERROR_NO_MEMORY;
    }

    out->shape = (int*) metadata;
    out->strides = (int*) &metadata[ndim * sizeof *out->shape];
    return out->data ? 0 : -1;
}

static void tensor_calculate_strides(const Tensor* out) {
    out->strides[out->ndim - 1] = 1;
    for (int i = out->ndim - 2; i >= 0; i--) {
        out->strides[i] = out->shape[i+1] * out->strides[i + 1];
    }
}

static int tensor_alloc(Tensor* out, const int* shape, const int ndim) {
    const int flat_length = tensor_flat_length(shape, ndim);

    if (tensor_alloc_data(out,flat_length) < 0) return -1;

    if (tensor_alloc_metadata(out,shape,ndim) < 0) return -1;

    out->ndim = ndim;
    out->length = flat_length;

    return 0;
}

TensorError tensor_empty(Tensor* out, const int* shape, const int ndim) {
    if (tensor_alloc(out, shape, ndim) < 0) {
        tensor_free(out);
        return TENSOR_ERROR_NO_MEMORY;
    }    memcpy(out->shape, shape, ndim * sizeof *out->shape);
    tensor_calculate_strides(out);
    return TENSOR_ERROR_NONE;
}

TensorError tensor_from_data(Tensor* out, const float* data, const int* shape,const int ndim) {
    if (tensor_alloc(out, shape, ndim) < 0) {
        tensor_free(out);
        return TENSOR_ERROR_NO_MEMORY;
    }
    memcpy(out->shape, shape, ndim * sizeof *out->shape);
    memcpy(out->data, data, out->length * sizeof *out->data);
    tensor_calculate_strides(out);
    return TENSOR_ERROR_NONE;
}

TensorError tensor_zeros(Tensor* out, const int* shape, const int ndim) {
    if (tensor_alloc(out, shape, ndim) < 0) {
        tensor_free(out);
        return TENSOR_ERROR_NO_MEMORY;
    }
    memcpy(out->shape, shape, ndim * sizeof *out->shape);
    tensor_calculate_strides(out);
    for (int i = 0; i < out->length; i++) {
        out->data[i] = 0.0f;
    }
    return TENSOR_ERROR_NONE;
}

TensorError tensor_ones(Tensor* out, const int* shape, const int ndim) {
    if (tensor_alloc(out, shape, ndim) < 0) {
        tensor_free(out);
        return TENSOR_ERROR_NO_MEMORY;
    }
    memcpy(out->shape, shape, ndim * sizeof *out->shape);
    tensor_calculate_strides(out);
    for (int i = 0; i < out->length; i++) {
        out->data[i] = 1.0f;
    }
    return TENSOR_ERROR_NONE;
}

TensorError tensor_fill(Tensor* out, const float num, const int* shape,int ndim) {
    if (tensor_alloc(out, shape, ndim) < 0) {
        tensor_free(out);
        return TENSOR_ERROR_NO_MEMORY;
    }
    memcpy(out->shape, shape, ndim * sizeof *out->shape);
    tensor_calculate_strides(out);
    for (int i = 0; i < out->length; i++) {
        out->data[i] = num;
    }
    return TENSOR_ERROR_NONE;
}

void tensor_free(Tensor* tensor) {
    free(tensor->data);
    free(tensor->shape);
    free(tensor);
};

void tensor_view_free(Tensor* tensor) {
    free(tensor->shape);
    free(tensor);
}


TensorError tensor_expand(Tensor* out, const Tensor* in, const int* new_shape, const int new_ndims) {

}

TensorError tensor_promote_to_col(Tensor* out, const Tensor* in);

const char* tensor_to_string(const Tensor* tensor);
const char* tensor_metadata_to_string(const Tensor* tensor);
const char* tensor_error_to_string(TensorError error);


