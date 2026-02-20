#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"


//

int main(){
	Tensor a;
	Tensor b;
	Tensor res;


	const float a_data[] = {1,2,3,4,5,6};
	const int a_shape[] = {2,3};
	const int a_ndim = 2;

	const float b_data[] = {1,2,3};
	const int b_shape[] = {3,1};
	const int b_ndim = 2;

	TensorError err = tensor_init_from_data(&a, a_data, a_shape,a_ndim);

	err = tensor_init_from_data(&b,b_data,b_shape,b_ndim);

	err = tensor_mat_mul(&res, &a, &b);

	printf("%s\n", tensor_error_to_string(err));

	char* a_buff = tensor_to_string(&a);
	printf("%s\n",a_buff);

	char* b_buff = tensor_to_string(&b);
	printf("%s\n",b_buff);

	char* res_buff = tensor_to_string(&res);
	printf("%s\n",res_buff);
	
	
	free(a_buff);
	free(b_buff);
	free(res_buff);

	tensor_destroy(&a);
	tensor_destroy(&b);
	tensor_destroy(&res);
}