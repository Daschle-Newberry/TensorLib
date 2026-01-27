#include <stddef.h>
#include <stdlib.h>

#include "tensor.h"

int main(){
    Tensor a;
    double a_data[] = {1.111111,2.0,3.0,4.0};
    size_t a_shape[] = {2,2};
    size_t a_ndim = 2;

    TensorError err = tensor_init_ones(&a,a_shape, a_ndim,T_TYPE_DOUBLE);

    printf("%s\n", tensor_error_to_string(err));

    printf("%.6g",((double*)a.data)[1]);
    char* buff = tensor_to_string(&a);
    printf("%s\n",buff);
    free(buff);

    tensor_destroy(&a);
    return 0;
}