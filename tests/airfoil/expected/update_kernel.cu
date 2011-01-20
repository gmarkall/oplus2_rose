#include "user_defined_types.h"
#include "op_datatypes.h"
#include "kernels.h"
__device__
#include <update.h>
__global__

void op_cuda_update(float *arg0,float *arg1,float *arg2,float *arg3,float *arg4,int set_size,void *block_reduct4)
{
  float arg4_l[1];
  for (int d = 0; d < 1; ++d) {
    arg4_l[d] = 0;
  }
  for (int n = threadIdx.x + blockIdx.x * blockDim.x; n < set_size; n += blockDim.x * gridDim.x) {
    update(arg0 + n * 4,arg1 + n * 4,arg2 + n * 4,arg3 + n * 1,arg4_l);
  }
  for (int d = 0; d < 1; ++d) {
    op_reduction2_1<OP_INC>(arg4 + d,arg4_l[d],block_reduct4);
  }
}

__global__

void op_cuda_update_reduction(int gridsize,float *arg4,void *block_reduct4)
{
  for (int d = 0; d < 1; ++d) {
    op_reduction2_2<OP_INC>(arg4 + d,block_reduct4,gridsize);
  }
}


float op_par_loop_update(const char *name,op_set set,struct op_dat<void> *arg0,int idx0,op_ptr *ptr0,enum op_access acc0,struct op_dat<void> *arg1,int idx1,op_ptr *ptr1,enum op_access acc1,struct op_dat<void> *arg2,int idx2,op_ptr *ptr2,enum op_access acc2,struct op_dat<void> *arg3,int idx3,op_ptr *ptr3,enum op_access acc3,struct op_dat<void> *arg4,int idx4,op_ptr *ptr4,enum op_access acc4)
{
  int bsize = BSIZE;
  int gridsize = (set.size - 1) / bsize + 1;
  int reduct_bytes = 0;
  int reduct_size = 0;
  reduct_bytes += ROUND_UP(1 * sizeof(float ));
  reduct_size = MAX(reduct_size,sizeof(float ));
  int reduct_shared = reduct_size * (BSIZE / 2);
  reallocReductArrays(reduct_bytes);
  reduct_bytes = 0;
  push_op_dat_as_reduct(*arg4,reduct_bytes);
  reduct_bytes += ROUND_UP(1 * sizeof(float ));
  mvReductArraysToDevice(reduct_bytes);
  void *block_reduct4 = 0;
  cudaMalloc(&block_reduct4,gridsize * sizeof(float ));
  int const_bytes = 0;
cudaEvent_t start, stop;
  float elapsed_time_ms = 0.00000F;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  op_cuda_update<<<gridsize,bsize,reduct_shared>>>(((float *)arg0->dat_d),((float *)arg1->dat_d),((float *)arg2->dat_d),((float *)arg3->dat_d),((float *)arg4->dat_d),set.size,block_reduct4);
  op_cuda_update_reduction<<<1,1,reduct_shared>>>(gridsize,((float *)arg4->dat_d),block_reduct4);
  cudaEventRecord(stop,0);
  cudaThreadSynchronize();
  cudaEventElapsedTime(&elapsed_time_ms,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  mvReductArraysToHost(reduct_bytes);
  pop_op_dat_as_reduct(*arg4);
  cudaFree(block_reduct4);
  return elapsed_time_ms;
}

