#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "tensor.h"
#include "tensor_operations.h"

int main(void) {
    Tensor a;
    Tensor b;
    Tensor res;

    float data_a[3] = {10,20,30};
    int shape_a[1] = {1};
    int ndim_a = 1;

    float data_b[8] = {100,101,102,103,200,201,202,203};
    int shape_b[3] = {2,1,4};
    int ndim_b = 3;

    TensorError err = tensor_init(&a, data_a, shape_a, ndim_a);
    printf("A init error: %s\n", error_to_string(err));
    // printf("A %s", tensor_metadata_to_string(&a));

    err = tensor_init(&b, data_b, shape_b, ndim_b);
    printf("B init error: %s\n", error_to_string(err));
    // printf("B %s", tensor_metadata_to_string(&b));

    TensorOperationError op_err = mul_tensor(&res, &a, &b);

    printf("operation error: %d\n", op_err);

    printf("%s\n",tensor_to_string(&res));




    return 0;
}