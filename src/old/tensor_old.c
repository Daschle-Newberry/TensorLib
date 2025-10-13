#include <stdlib.h>
#include <string.h>

#include <stdbool.h>
#include <stdio.h>

#include "string_builder.h"


static void calculate_strides(int* strides_out, const int* shape, const int ndim) {
    strides_out[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides_out[i] = shape[i+1] * strides_out[i + 1];
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


TensorError tensor_init(Tensor* out,const float* data,const int* shape,const int ndim) {
    int flat_length = 1;
    for (int i = 0; i < ndim; i++) {
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
    out->shape = (int*) metadata;
    out->strides = (int*) &metadata[ndim * sizeof *out->shape];

    memcpy(out->shape, shape, ndim * sizeof *out->shape);

    out->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        out->strides[i] = out->shape[i+1] * out->strides[i + 1];
    }

    if (data) {
        memcpy(out->data, data, flat_length * sizeof *out->data);
    }

    return TENSOR_ERROR_NONE;
}


TensorError tensor_expand(Tensor* tensor, const int new_dims) {
    const int added_dims = new_dims - tensor->ndim;

    if (added_dims < 0) return TENSOR_ERROR_NEGATIVE_DIM;
    if (added_dims == 0) return TENSOR_ERROR_NONE;

    //Allocate memory for the new shape and strides
    char* metadata = malloc(
           (new_dims * sizeof *tensor->shape) + (new_dims * sizeof *tensor->strides)
        );
    if (metadata == NULL) {
        return TENSOR_ERROR_NO_MEMORY;
    }

    int* new_shape = (int*) metadata;
    int* new_strides = (int*) &metadata[new_dims * sizeof *tensor->shape];

    for (int i = 0; i < added_dims; i++) {
        new_shape[i] = 1;
    }
    for (int i = 0; i < tensor->ndim; i++) {
        new_shape[added_dims + i] = tensor->shape[i];
    }

    calculate_strides(new_strides, new_shape,new_dims);

    free(tensor->shape);

    tensor->shape = new_shape;
    tensor->strides = new_strides;
    tensor->ndim = new_dims;


    return TENSOR_ERROR_NONE;
}

TensorError tensor_broadcast_to(const Tensor* tensor, const int* shape, int ndim, const bool mat_aware) {
    if (tensor->ndim != ndim) return TENSOR_ERROR_INPUT_DIM_MISMATCH;
    if (mat_aware){ndim -= 2;}
    for (int i = 0; i < ndim; i++) {
        if (shape[i] != tensor->shape[i]) {
            tensor->shape[i] = shape[i];
            tensor->strides[i] = 0;
        }
    }

    return TENSOR_ERROR_NONE;
}


TensorError tensor_shallow_copy(Tensor* out, const Tensor* original, const int new_dims) {
    out->data = original->data;

    const int added_dims = new_dims - original->ndim;

    if (added_dims < 0) return TENSOR_ERROR_NEGATIVE_DIM;
    //Allocate memory for the new shape and strides
    char* metadata = malloc(
           (new_dims * sizeof *original->shape) + (new_dims * sizeof *original->strides)
        );
    if (metadata == NULL) {
        return TENSOR_ERROR_NO_MEMORY;
    }

    int* new_shape = (int*) metadata;
    int* new_strides = (int*) &metadata[new_dims * sizeof *original->shape];

    for (int i = 0; i < added_dims; i++) {
        new_shape[i] = 1;
    }
    for (int i = 0; i < original->ndim; i++) {
        new_shape[added_dims + i] = original->shape[i];
    }

    calculate_strides(new_strides, new_shape,new_dims);

    out->shape = new_shape;
    out->strides = new_strides;
    out->ndim = new_dims;


    return TENSOR_ERROR_NONE;
}


TensorError tensor_promote_to_col(const Tensor* tensor) {
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
    sb_append(&sb, "]\n");
    sb_append(&sb, "--------------------------------\n");

    return sb.buff;
}

