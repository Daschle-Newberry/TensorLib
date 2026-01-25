
#include <stddef.h>
#include "tensor.h"

int main(){
    Tensor a;
    size_t a_shape[] = {2,2};
    size_t a_ndim = 2;

    TensorError err = tensor_init_empty(&a, a_shape, a_ndim,T_TYPE_FLOAT);
}