#include "tensor_format.h"
#include "tensor.h"

// T_TYPE_INT,
//     T_TYPE_FLOAT,
//     T_TYPE_DOUBLE,
//     T_TYPE_LONG,

void format_int(char* buff, size_t buff_size, const void* data, size_t index){
	snprintf(buff, buff_size,"%d",((int*)data)[index]);
}

void format_float(char* buff, size_t buff_size, const void* data, size_t index){
	snprintf(buff, buff_size,"%.6g",((float*)data)[index]);
}

void format_double(char* buff, size_t buff_size, const void* data, size_t index){
	snprintf(buff, buff_size,"%.6g",((double*)data)[index]);
}

void format_long(char* buff, size_t buff_size, const void* data, size_t index){
	snprintf(buff, buff_size,"%ld",((long*)data)[index]);
}
TensorFormatFunc tensor_get_data_formatter(TensorType type){
	switch(type){
		case T_TYPE_INT: return format_int;
		case T_TYPE_FLOAT: return format_float;
		case T_TYPE_DOUBLE: return format_double;
		case T_TYPE_LONG: return format_long;
	}
}

