#include <stdlib.h>
#include <string.h>

#include "tensor.h"

#include <stdio.h>

#include "string_builder.h"

static const char* TensorErrorStrings[] = {
    [TENSOR_ERROR_NONE] = "TENSOR_ERROR_NONE",
    [TENSOR_ERROR_NO_MEMORY] = "TENSOR_ERROR_NO_MEMORY",
    [TENSOR_ERROR_INPUT_DIM_MISMATCH] = "TENSOR_ERROR_INPUT_DIM_MISMATCH",
    [TENSOR_ERROR_COUNT] = "TENSOR_ERROR_COUNT",
};


TensorError tensor_init(Tensor* out,const float* data,const size_t* shape,const size_t ndim) {
    size_t flat_length = 1;
    for (size_t i = 0; i < ndim; i++) {
        flat_length *= shape[i];
    }

    out->data = malloc(
    (flat_length * sizeof *out->data)
    );
    if (out->data == NULL) {
        return TENSOR_ERROR_NO_MEMORY;
    }

    char* metadata = malloc(
       (ndim * sizeof *out->shape) + (ndim * sizeof *out->strides)
    );
    if (metadata == NULL) {
        return TENSOR_ERROR_NO_MEMORY;
    }

    out->ndim = ndim;
    out->shape = (size_t*) metadata;
    out->strides = (size_t*) &metadata[ndim * sizeof *out->shape];

    memcpy(out->shape, shape, ndim * sizeof *out->shape);

    out->strides[ndim - 1] = 1;
    for (ssize_t i = (ssize_t)ndim - 2; i >= 0; i--) {
        out->strides[i] = out->shape[i+1] * out->strides[i + 1];
    }

    if (data) {
        memcpy(out->data, data, flat_length * sizeof *out->data);
    }

    return TENSOR_ERROR_NONE;
}


TensorError tensor_shallow_copy(Tensor* out, const Tensor* original) {
    out->data = original->data;
    out->shape = original->shape;
    out->strides = original->strides;
    out->ndim = original->ndim;

    return TENSOR_ERROR_NONE;
}

TensorError tensor_reshape(Tensor* out, float* data, const size_t* shape, const size_t ndim) {
    out->data = data;

    char* metadata = malloc(
      (ndim * sizeof *out->shape) + (ndim * sizeof *out->strides)
    );

    if (metadata == NULL) {
        return TENSOR_ERROR_NO_MEMORY;
    }

    out->ndim = ndim;
    out->shape = (size_t*) metadata;
    out->strides = (size_t*) &metadata[ndim * sizeof *out->shape];

    memcpy(out->shape, shape, ndim * sizeof *out->shape);

    out->strides[ndim - 1] = 1;
    for (ssize_t i = (ssize_t)ndim - 2; i >= 0; i--) {
        out->strides[i] = out->shape[i+1] * out->strides[i + 1];
    }

    return TENSOR_ERROR_NONE;
}

void build_string(StringBuilder* sb, const Tensor* tensor, size_t offset, size_t dim) {
    if (dim == tensor->ndim - 1) {
        sb_append(sb, "     [");
        for (size_t i = 0; i <tensor->shape[dim]; i++) {
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
    for (size_t i = 0; i < tensor->shape[dim]; i++) {
        size_t new_offset = offset + i * tensor->strides[dim];
        build_string(sb, tensor, new_offset, dim + 1);
    }
    sb_append(sb,"]\n");
}

char* tensor_to_string(const Tensor* tensor) {
    StringBuilder sb;
    init_sb(&sb);
    build_string(&sb,tensor, 0, 0);
    return sb.buff;
}

char* tensor_metadata_to_string(const Tensor* tensor) {
    StringBuilder sb;
    init_sb(&sb);

    sb_append(&sb, "----------- METADATA -----------\n");
    char ndim_buff[21];
    snprintf(ndim_buff, sizeof(ndim_buff),"%llu",tensor->ndim);

    sb_append(&sb, "ndim: ");
    sb_append(&sb, ndim_buff);
    sb_append(&sb, "\nShape: [");

    for (size_t i = 0; i < tensor->ndim; i++) {
        char shape_buff[21];
        snprintf(shape_buff, sizeof(shape_buff),"%llu",tensor->shape[i]);

        sb_append(&sb, shape_buff);
        if (i < tensor->ndim - 1) {
            sb_append(&sb, ", ");
        }
    }

    sb_append(&sb, "]\nStrides: [");

    for (size_t i = 0; i < tensor->ndim; i++) {
        char stride_buff[21];
        snprintf(stride_buff, sizeof(stride_buff),"%llu",tensor->strides[i]);
        sb_append(&sb, stride_buff);
        if (i < tensor->ndim - 1) {
            sb_append(&sb, ", ");
        }
    }
    sb_append(&sb, "]\n");
    sb_append(&sb, "--------------------------------\n");

    return sb.buff;
}

char* error_to_string(const TensorError code) {
    return TensorErrorStrings[code];
}
