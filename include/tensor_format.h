#ifndef TENSOR_FORMAT_H
#define TENSOR_FORMAT_H

#include <stddef.h>
#include "tensor.h"

typedef void (*TensorFormatFunc)(
		char* buff,
		size_t buff_size,
		const void* data,
		size_t index
	);

TensorFormatFunc tensor_get_data_formatter(TensorType type);

#endif //TENSOR_FORMAT_H