#ifndef TENSOR_H
#define TENSOR_H

typedef enum {
    TENSOR_ERROR_NONE,
    TENSOR_ERROR_NO_MEMORY,
    TENSOR_ERROR_INPUT_DIM_MISMATCH,
    TENSOR_ERROR_COUNT
}TensorError;

typedef struct {
    size_t ndim;
    size_t* shape;
    size_t* strides;
    float* data;
} Tensor;

TensorError tensor_init(Tensor* out,const float* data,const size_t* shape,size_t ndim);
TensorError tensor_init_zero(Tensor* out,const size_t* shape,size_t ndim);

TensorError tensor_shallow_copy(Tensor* out, const Tensor* original);
TensorError tensor_reshape(Tensor* out, float* data, const size_t* shape, size_t ndim);

char* tensor_to_string(const Tensor* tensor);
char* tensor_metadata_to_string(const Tensor* tensor);
char* error_to_string(TensorError code);

#endif //TENSOR_H