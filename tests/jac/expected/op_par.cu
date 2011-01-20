#include "op_lib.cu"
#import "op_datatypes.cpp"
__constant__
float alpha[2UL];
#include "res_kernel.cu"
#include "update_kernel.cu"
