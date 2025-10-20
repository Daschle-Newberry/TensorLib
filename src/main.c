#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "tensor.h"
int main(void) {
    Tensor out;
    Tensor a;
    Tensor b;

    float data_a[] = {1,2,3};
    int shape_a[] = {3,1};
    int ndim_a = 2;

    TensorError err = tensor_from_data(&a, data_a, shape_a, ndim_a);
    printf("A init error: %s\n", tensor_error_to_string(err));
    printf("A\n %s", tensor_metadata_to_string(&a));
    printf("%s\n", tensor_to_string(&a));


    err = tensor_expand(&out, &a, (int[]){3,6}, 2);
    printf("Out error: %s\n", tensor_error_to_string(err));
    printf("Out\n %s", tensor_metadata_to_string(&out));
    printf("%s\n", tensor_to_string(&out));

    printf("Press enter to exit...");
    getchar();
    return 0;
}