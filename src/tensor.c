#include <stdlib.h>
#include <string.h>

#include "tensor.h"

#include <stdio.h>

#include "string_builder.h"

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

static void build_string(StringBuilder* sb, const Tensor* tensor, const int offset, const int dim) {
    if (dim == tensor->ndim - 1) {
        sb_append(sb, "     [");
        for (int i = 0; i <tensor->shape[dim]; i++) {
            char buff[32];
            snprintf(buff, sizeof(buff),"%.6g",tensor->data[offset + i]);
            sb_append(sb,buff);

            if (i < tensor->shape[dim]-1)
                sb_append(sb,", ");
        }
        sb_append(sb,"]\n");
        return;
    }

    sb_append(sb,"[\n");
    for (int i = 0; i < tensor->shape[dim]; i++) {
        const int new_offset = offset + i * tensor->strides[dim];
        build_string(sb, tensor, new_offset, dim + 1);
    }
    sb_append(sb,"]\n");
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

TensorError tensor_promote_to_col(Tensor* out, const Tensor* in) {
    const int m = tensor->shape[tensor->ndim - 2];
    const int n = tensor->shape[tensor->ndim -1];

    const int ms = tensor->strides[tensor->ndim - 2];
    const int ns = tensor->strides[tensor->ndim - 1];

    tensor->shape[tensor->ndim - 2] = n;
    tensor->shape[tensor->ndim - 1] = m;
    tensor->strides[tensor->ndim - 2] = ns;
    tensor->strides[tensor->ndim - 1] = ms;

    return TENSOR_ERROR_NONE;
}

const char* tensor_to_string(const Tensor* tensor) {
    StringBuilder sb;
    init_sb(&sb);
    build_string(&sb,tensor, 0, 0);
    return sb.buff;
}

const char* tensor_metadata_to_string(const Tensor* tensor) {
    StringBuilder sb;
    init_sb(&sb);

    sb_append(&sb, "----------- METADATA -----------\n");
    char ndim_buff[21];
    snprintf(ndim_buff, sizeof(ndim_buff),"%d",tensor->ndim);

    sb_append(&sb, "ndim: ");
    sb_append(&sb, ndim_buff);
    sb_append(&sb, "\nShape: [");

    for (int i = 0; i < tensor->ndim; i++) {
        char shape_buff[21];
        snprintf(shape_buff, sizeof(shape_buff),"%d",tensor->shape[i]);

        sb_append(&sb, shape_buff);
        if (i < tensor->ndim - 1) {
            sb_append(&sb, ", ");
        }
    }

    sb_append(&sb, "]\nStrides: [");

    for (int i = 0; i < tensor->ndim; i++) {
        char stride_buff[21];
        snprintf(stride_buff, sizeof(stride_buff),"%d",tensor->strides[i]);
        sb_append(&sb, stride_buff);
        if (i < tensor->ndim - 1) {
            sb_append(&sb, ", ");
        }
    }
}

const char* tensor_error_to_string(const TensorError error){return TensorErrorStrings[error];}


