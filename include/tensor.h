#ifndef TENSOR_H
#define TENSOR_H
typedef enum {
    TENSOR_ERROR_NONE,
    TENSOR_ERROR_NO_MEMORY,
    TENSOR_ERROR_INPUT_DIM_MISMATCH,
    TENSOR_ERROR_NEGATIVE_DIM,
    TENSOR_ERROR_COUNT
}TensorError;

typedef struct {
    int ndim;
    int* shape;
    int* strides;
    float* data;
} Tensor;

TensorError tensor_init(Tensor* out,const float* data,const int* shape,int ndim);
TensorError tensor_init_zero(Tensor* out,const int* shape,int ndim);

TensorError tensor_expand_dims(Tensor* tensor, int new_dims);
TensorError tensor_shallow_copy(Tensor* out, const Tensor* original, int new_dims);
TensorError tensor_reshape(Tensor* out, float* data, const int* shape, int ndim);
TensorError tensor_broadcast_to(const Tensor* tensor, const int* shape, int ndim);

const char* tensor_to_string(const Tensor* tensor);
const char* tensor_metadata_to_string(const Tensor* tensor);
const char* tensor_error_to_string(TensorError code);

#endif //TENSOR_H