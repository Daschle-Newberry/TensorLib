#include <stdio.h>
#include <time.h>
#include "tensor.h"

int main(){
	Tensor out;
	Tensor a;
	int N = 1000;
	int a_shape[] = {1,N};
	int a_ndim = 2;
	Tensor b;
	int b_shape[] = {N,1};
	int b_ndim = 2;

	TensorError err = tensor_init_empty(&a,a_shape,a_ndim);

	err = tensor_init_empty(&b,b_shape,b_ndim);

	// printf("%s",tensor_metadata_to_string(&a));
	// printf("%s",tensor_metadata_to_string(&b));


	err = tensor_mat_mul(&out, &a, &b);
	printf("%s\n", tensor_error_to_string(err));
	printf("%s\n",tensor_metadata_to_string(&out));

	return 0;
}