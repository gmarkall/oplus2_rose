#ifndef OP_KERNELS
#define OP_KERNELS
#include "op_datatypes.h"
float op_par_loop_res(const char *name,op_set set,struct op_dat<void> *arg0,int idx0,op_ptr *ptr0,enum op_access acc0,struct op_dat<void> *arg1,int idx1,op_ptr *ptr1,enum op_access acc1,struct op_dat<void> *arg2,int idx2,op_ptr *ptr2,enum op_access acc2,struct op_dat<void> *arg3,int idx3,op_ptr *ptr3,enum op_access acc3);
float op_par_loop_update(const char *name,op_set set,struct op_dat<void> *arg0,int idx0,op_ptr *ptr0,enum op_access acc0,struct op_dat<void> *arg1,int idx1,op_ptr *ptr1,enum op_access acc1,struct op_dat<void> *arg2,int idx2,op_ptr *ptr2,enum op_access acc2,struct op_dat<void> *arg3,int idx3,op_ptr *ptr3,enum op_access acc3,struct op_dat<void> *arg4,int idx4,op_ptr *ptr4,enum op_access acc4);
#endif
