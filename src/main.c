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

    float data_a[27] = {1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9};
    size_t shape_a[3] = {3,3,3};
    size_t ndim_a = 3;

    float data_b[9] = {1,2,3,4,5,6,7,8,9};
    size_t shape_b[3] = {3,3};
    size_t ndim_b = 3;

    TensorError err = tensor_init(&a, data_a, shape_a, ndim_a);
    printf("A init error: %s\n", error_to_string(err));

    err = tensor_init(&b, data_b, shape_b, ndim_b);
    printf("B init error: %s\n", error_to_string(err));

    TensorOperationError op_err = mul_tensor(&res, &a, &b);

    printf("operation error: %d\n", op_err);

    printf("%s\n",tensor_metadata_to_string(
        res));




    return 0;
}