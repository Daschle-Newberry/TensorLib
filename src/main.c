#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "tensor.h"





int main(void) {
    Tensor out;
    Tensor a;
    Tensor b;


    float data_a[] = {2,2,2,2,2,2,2,2,2};
    int shape_a[2] = {3,3};
    int ndim_a = 2;

    float data_b[6] = {1,1,1,1,1,1};
    int shape_b[2] = {2,3};
    int ndim_b = 2;




    TensorError err = tensor_from_data(&a, data_a, shape_a, ndim_a);
    // printf("A init error: %s\n", tensor_error_to_string(err));
    // printf("A\n %s", tensor_metadata_to_string(&a));

    printf("%s\n",tensor_error_to_string(err));

    printf("%s\n", tensor_to_string(&a));


    return 0;
}