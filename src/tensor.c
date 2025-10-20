#include <stdlib.h>
#include <string.h>

#include "tensor.h"

#include <stdio.h>

#include "string_builder.h"

static const char* TensorErrorStrings[] = {
    [TENSOR_ERROR_NONE] = "TENSOR_ERROR_NONE",
    [TENSOR_ERROR_NO_MEMORY] = "TENSOR_ERROR_NO_MEMORY",
    [TENSOR_ERROR_INVALID_ARGUMENT] = "TENSOR_ERROR_INVALID_ARGUMENT",
    [TENSOR_ERROR_INPUT_DIM_MISMATCH] = "TENSOR_ERROR_INPUT_DIM_MISMATCH",
    [TENSOR_ERROR_NEGATIVE_DIM] = "TENSOR_ERROR_NEGATIVE_DIM",
    [TENSOR_ERROR_CANNOT_BROADCAST] = "TENSOR_ERROR_CANNOT_BROADCAST",
    [TENSOR_ERROR_CANNOT_EXPAND] = "TENSOR_ERROR_CANNOT_EXPAND"
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

//[2,2,2]
/**
 *[
 *  [
 *      [1,2],
 *      [3,4]
 *  ],
 *  [
 *      [5,6],
 *      [7,8]
 *  ]
 *]
 */
static void build_string_2(StringBuilder* sb, const Tensor* tensor, const int offset, const int dim, const int indent_level) {
    for (int i = 0; i < indent_level; i++) sb_append(sb, "  ");
    sb_append(sb,"[");

    for (int i = 0; i < tensor->shape[dim]; i++) {
        int current_offset = offset + i * tensor->strides[dim];

        if (dim == tensor->ndim - 1) {
            char buff[32];
            snprintf(buff, sizeof(buff),"%.6g",tensor->data[current_offset]);
            sb_append(sb,buff);
        }else {
            sb_append(sb, "\n");
            build_string_2(sb,tensor,current_offset, dim + 1, indent_level + 1);
        }

        if (i < tensor->shape[dim] - 1) sb_append(sb, ", ");
    }

    if (dim != tensor->ndim - 1) {
        sb_append(sb,"\n");
        for (int i = 0; i < indent_level; i++) sb_append(sb, "  ");
    }
    sb_append(sb,"]");

}
static void build_string(StringBuilder* sb, const Tensor* tensor, const int offset, const int dim) {
    if (dim == tensor->ndim - 1) {
        sb_append(sb, "     [");
        for (int i = 0; i <tensor->shape[dim]; i++) {
            char buff[32];
            snprintf(buff, sizeof(buff),"%.6g",tensor->data[offset + i * tensor->strides[dim]]);
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
    }

    memcpy(out->shape, shape, ndim * sizeof *out->shape);
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

float tensor_get(const Tensor* tensor, const int* idx) {
    int offset=  0;
    for (int i = 0; i < tensor->ndim; i++) {
        offset += idx[i] * tensor->strides[i];
    }
    return tensor->data[offset];
}
TensorError tensor_expand(Tensor* out, const Tensor* in, const int* new_shape, const int new_ndim) {
    if (in->ndim > new_ndim) return TENSOR_ERROR_INVALID_ARGUMENT;
    if (tensor_alloc_view(out, in, new_ndim) < 0) {
        tensor_view_free(out);
        return TENSOR_ERROR_NO_MEMORY;
    }

    const int diff = new_ndim - in->ndim;

    //ALl new dims
    memcpy(out->shape,new_shape, diff * sizeof (int));

    for (int i = 0; i < diff; i++) {
        out->strides[i] = 0;
    }

    //Overlapping dims
    for (int i = diff; i < new_ndim; i++) {
        int new = new_shape[i];
        int old = in->shape[i - diff];

        if (new == old) {
            out->strides[i] = in->strides[i - diff];
        }
        else if (new == 1 || old == 1) {
            out->strides[i] = 0;
        }else {
            tensor_view_free(out);
            return TENSOR_ERROR_CANNOT_EXPAND;
        }

        out->shape[i] = new_shape[i];
    }

    out->data = in->data;

    return TENSOR_ERROR_NONE;
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
    build_string_2(&sb,tensor, 0, 0,0);
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


