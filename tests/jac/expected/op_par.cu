#define OP_KERNELS_MAX 2
#include <op_lib.cu>
#include <user_defined_types.h>
#import <op_datatypes.cpp>
__constant__
float alpha[2UL];
#include "res_kernel.cu"
#include "update_kernel.cu"
