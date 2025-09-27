#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "tensor.h"
#include "tensor_operations.h"

int main(void) {
    Tensor a;
    Tensor b;
    int size = 100;
    int data_length = size*size;
    float data_a[data_length];
    int shape_a[2] = {size,size};
    int ndim_a = 2;

    float data_b[data_length];
    int shape_b[2] = {size,size};
    int ndim_b = 2;

    for (int i = 0; i < data_length; i++) {
        data_a[i] = (float)i;
        data_b[i] = (float)i;
    }


    TensorError err = tensor_init(&a, data_a, shape_a, ndim_a);
    printf("A init error: %s\n", tensor_error_to_string(err));
    printf("A\n %s", tensor_metadata_to_string(&a));

    err = tensor_init(&b, data_b, shape_b, ndim_b);
    printf("B init error: %s\n", tensor_error_to_string(err));
    printf("B\n %s", tensor_metadata_to_string(&b));


    float count = 5000;

    float i = 0;


    clock_t start = clock();
    while (i < count) {
        Tensor res;
        mul_tensor(&res, &a, &b);
        i++;
    }

    clock_t end = clock();

    double total_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%f operations per second", count/total_time);
    return 0;
}