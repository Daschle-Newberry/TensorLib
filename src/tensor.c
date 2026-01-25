#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "tensor.h"
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

static const size_t TypeBytes[] = {
    [T_TYPE_INT] = sizeof(int),
    [T_TYPE_FLOAT] = sizeof(float),
    [T_TYPE_DOUBLE] = sizeof(double),
    [T_TYPE_LONG] = sizeof(long)
};

static size_t compute_flat_length(const size_t* shape, int ndim) {
    size_t flat_length = 1;
    for (size_t i = 0; i < ndim; i++) {
        flat_length *= shape[i];
    }
    return flat_length;
}

static int tensor_alloc(Tensor* out) {
    out->data = malloc(out->length * out->dbytes);

    size_t* metadata = malloc(
       2 * (out->ndim * sizeof *out->strides)
    );

    out->shape = metadata;
    out->strides = &metadata[out->ndim];

    int status = 0;
    if(!out->data){
        status = -1;
        free(out->data);
    }
    if(!metadata){
        status = -1;
        free(out->data);
    }

    return status;
}

static int tensor_alloc_view(Tensor* out) {
    size_t* metadata = malloc(
       2 * (out->ndim * sizeof *out->strides)
    );

    out->shape = metadata;
    out->strides = &metadata[out->ndim];

    if(!metadata) return -1;
    return 0;
}

static void tensor_generate_strides(Tensor* out) {
    out->strides[out->ndim - 1] = 1;
    for (size_t i = out->ndim - 2; i >= 0; i--) {
        out->strides[i] = out->shape[i+1] * out->strides[i + 1] * out->dbytes;
    }
}

static void build_string(StringBuilder* sb, const Tensor* tensor, const int offset, const int dim, const int indent_level) {
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
            build_string(sb,tensor,current_offset, dim + 1, indent_level + 1);
        }

        if (i < tensor->shape[dim] - 1) sb_append(sb, ", ");
    }

    if (dim != tensor->ndim - 1) {
        sb_append(sb,"\n");
        for (int i = 0; i < indent_level; i++) sb_append(sb, "  ");
    }
    sb_append(sb,"]");

}

TensorError tensor_init_empty(Tensor* out, const size_t* shape, size_t ndim, TensorType dtype) {
    out->dbytes = TypeBytes[dtype];
    out->dtype = dtype;
    out->length = compute_flat_length(shape, ndim);
    out->ndim = ndim;

    if (tensor_alloc(out) < 0) return TENSOR_ERROR_NO_MEMORY;

    memcpy(out->shape, shape, ndim * sizeof *out->shape);
    tensor_generate_strides(out);
    return TENSOR_ERROR_NONE;
}

TensorError tensor_init_from_data(Tensor* out, const void* data, const size_t* shape, size_t ndim, TensorType dtype) {
    out->dbytes = TypeBytes[dtype];
    out->dtype = dtype;
    out->length = compute_flat_length(shape, ndim);
    out->ndim = ndim;

    if (tensor_alloc(out) < 0) return TENSOR_ERROR_NO_MEMORY;
    

    memcpy(out->shape, shape, ndim * sizeof *out->shape);
    memcpy(out->data, data, out->length * out->dbytes);
    tensor_generate_strides(out);

    return TENSOR_ERROR_NONE;
}

TensorError tensor_init_zeros(Tensor* out, const size_t* shape, size_t ndim, TensorType dtype) {
    out->dbytes = TypeBytes[dtype];
    out->dtype = dtype;
    out->length = compute_flat_length(shape, ndim);
    out->ndim = ndim;

    if (tensor_alloc(out) < 0) return TENSOR_ERROR_NO_MEMORY;

    memcpy(out->shape, shape, ndim * out->dbytes);
    tensor_generate_strides(out);

    memset(out->data,0, out->length * out->dbytes);

    return TENSOR_ERROR_NONE;
}

TensorError tensot_init_ones(Tensor* out, const size_t* shape, size_t ndim, TensorType dtype){
    out->dbytes = TypeBytes[dtype];
    out->dtype = dtype;
    out->length = compute_flat_length(shape, ndim);
    out->ndim = ndim;

    if (tensor_alloc(out) < 0) return TENSOR_ERROR_NO_MEMORY;

    memcpy(out->shape, shape, ndim * out->dbytes);
    tensor_generate_strides(out);
    
    //Can be simplified with macros
    switch(dtype){
        case T_TYPE_INT:{
            int* data = (int*)out->data;
            for(size_t i = 0; i < out->length; i++){
                data[i] = 0;
            }
            break;
        }

        case T_TYPE_FLOAT:{
            float* data = (float*)out->data;
            for(size_t i = 0; i < out->length; i++){
                data[i] = 0.0f;
            }
            break;
        }
        case T_TYPE_DOUBLE:{
            double* data = (double*)out->data;
            for(size_t i = 0; i < out->length; i++){
                data[i] = 0.0;
            }
            break;
        }
        case T_TYPE_LONG:{
            long* data = (long*)out->data;
            for(size_t i = 0; i < out->length; i++){
                data[i] = 0.0l;
            }
            break;
        }

    }

    return TENSOR_ERROR_NONE;
}

TensorError tensor_init_fill(Tensor* out, const void* num, const size_t* shape,size_t ndim, TensorType dtype) {
    out->dbytes = TypeBytes[dtype];
    out->dtype = dtype;
    out->length = compute_flat_length(shape, ndim);
    out->ndim = ndim;

    if (tensor_alloc(out) < 0) return TENSOR_ERROR_NO_MEMORY;

    memcpy(out->shape, shape, ndim * out->dbytes);
    tensor_generate_strides(out);
    
    //Can be simplified with macros
    switch(dtype){
        case T_TYPE_INT:{
            int* data = (int*)out->data;
            for(size_t i = 0; i < out->length; i++){
                data[i] = *(int*) num;
            }
            break;
        }

        case T_TYPE_FLOAT:{
            float* data = (float*)out->data;
            for(size_t i = 0; i < out->length; i++){
                data[i] = *(float*) num;
            }
            break;
        }
        case T_TYPE_DOUBLE:{
            double* data = (double*)out->data;
            for(size_t i = 0; i < out->length; i++){
                data[i] = *(double*) num;
            }
            break;
        }
        case T_TYPE_LONG:{
            long* data = (long*)out->data;
            for(size_t i = 0; i < out->length; i++){
                data[i] = *(long*) num;
            }
            break;
        }

    }

    return TENSOR_ERROR_NONE;
}


void tensor_destroy(Tensor* tensor) {
    free(tensor->data);
    free(tensor->shape);
};

void tensor_view_destroy(Tensor* tensor) {
    free(tensor->shape);
}

TensorError tensor_broadcast_to(Tensor* out, const Tensor* in, const size_t* new_shape, const size_t new_ndim) {
    if (in->ndim > new_ndim) return TENSOR_ERROR_INVALID_ARGUMENT;

    out->dbytes = in->dbytes;
    out->dtype = in->dtype;
    out->length = in->length;
    out->ndim = new_ndim;

    if (tensor_alloc_view(out) < 0) return TENSOR_ERROR_NO_MEMORY;

    const size_t diff = new_ndim - in->ndim;

    memcpy(out->shape,new_shape, diff * sizeof *out->shape);

    memset(out->strides, 0,diff * sizeof *out->shape);
  
    for (size_t i = diff; i < new_ndim; i++) {
        const size_t new = new_shape[i];
        const size_t old = in->shape[i - diff];

        if (new == old) {
            out->strides[i] = in->strides[i - diff];
        }
        else if (new == 1 || old == 1) {
            out->strides[i] = 0;
        }else {
            tensor_view_destroy(out);
            return TENSOR_ERROR_CANNOT_BROADCAST;
        }

        out->shape[i] = new_shape[i];
    }

    out->data = in->data;

    return TENSOR_ERROR_NONE;
}


TensorError tensor_matrix_broadcast_to(Tensor* out, const Tensor* in, const int* new_shape, const int new_ndim) {
    if (in->ndim > new_ndim) return TENSOR_ERROR_INVALID_ARGUMENT;
    if (tensor_alloc_view(out, in, new_ndim) < 0) {
        tensor_view_destroy(out);
        return TENSOR_ERROR_NO_MEMORY;
    }

    const int diff = new_ndim - in->ndim;

    memcpy(out->shape,new_shape, diff * sizeof (int));

    for (int i = 0; i < diff; i++) {
        out->strides[i] = 0;
    }

    for (int i = diff; i < new_ndim - 2; i++) {
        const int new = new_shape[i];
        const int old = in->shape[i - diff];

        if (new == old) {
            out->strides[i] = in->strides[i - diff];
        }
        else if (new == 1 || old == 1) {
            out->strides[i] = 0;
        }else {
            tensor_view_destroy(out);
            return TENSOR_ERROR_CANNOT_BROADCAST;
        }

        out->shape[i] = new_shape[i];
    }

    out->shape[out->ndim - 2] = in->shape[in->ndim - 2];
    out->shape[out->ndim - 1] = in->shape[in->ndim - 1];

    out->strides[out->ndim - 2] = in->strides[in->ndim - 2];
    out->strides[out->ndim - 1] = in->strides[in->ndim - 1];

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

TensorError tensor_promote_to_row(Tensor* out, const Tensor* in) {
    if (in->ndim > 1) return TENSOR_ERROR_INVALID_ARGUMENT;

    if (tensor_alloc_view(out, in, 2) < 0) return TENSOR_ERROR_NO_MEMORY;

    out->shape[0] = 1;
    out->shape[1] = in->shape[0];;

    out->strides[0] = 0;
    out->strides[1] = 1;

    out->data = in->data;

    return TENSOR_ERROR_NONE;
}

const char* tensor_to_string(const Tensor* tensor) {
    StringBuilder sb;
    init_sb(&sb);
    build_string(&sb,tensor,0, 0, 0);
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


