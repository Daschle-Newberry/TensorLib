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

static int tensor_alloc_metadata(Tensor* out, const int ndim) {
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

static int tensor_alloc(Tensor* out, const int* shape, const int ndim) {
    const int flat_length = tensor_flat_length(shape, ndim);

    if (tensor_alloc_data(out,flat_length) < 0) return -1;

    if (tensor_alloc_metadata(out,ndim) < 0) return -1;

    out->ndim = ndim;
    out->length = flat_length;

    return 0;
}

static int tensor_alloc_view(Tensor* out, const Tensor* in, const int ndim) {
    if (tensor_alloc_metadata(out,ndim) < 0) return -1;

    out->ndim = ndim;
    out->length = in->length;

    return 0;
}

static void tensor_calculate_strides(const Tensor* out) {
    out->strides[out->ndim - 1] = 1;
    for (int i = out->ndim - 2; i >= 0; i--) {
        out->strides[i] = out->shape[i+1] * out->strides[i + 1];
    }
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


TensorError tensor_expand(Tensor* out, const Tensor* in, const int* new_shape, const int new_ndim) {
    if (tensor_alloc_view(out, in, new_ndim) < 0) return TENSOR_ERROR_NO_MEMORY;

    for (int i = 0; i < in->ndim; i++) {
        if (new_shape[new_ndim - i] != in->shape[i]) return TENSOR_ERROR_CANNOT_EXPAND;
        out->shape[new_ndim - i] = new_shape[new_ndim - i];
    }

    for (int i = new_ndim - in->ndim; i < new_ndim; i++) {
        out->shape[i] = new_shape[i];
    }

    out->data = in->data;

}

TensorError tensor_promote_to_col(Tensor* out, const Tensor* in) {
    if (in->ndim > 1) return TENSOR_ERROR_INVALID_ARGUMENT;

    if (tensor_alloc_view(out, in, 2) < 0) return TENSOR_ERROR_NO_MEMORY;

    out->shape[0] = in->shape[0];
    out->shape[1] = 1;

    out->strides[0] = 1;
    out->strides[1] = 0;

    out->data = in->data;

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
    sb_append(&sb, "]\n");
    sb_append(&sb, "--------------------------------\n");

    return sb.buff;
}

const char* tensor_error_to_string(const TensorError error){return TensorErrorStrings[error];}


