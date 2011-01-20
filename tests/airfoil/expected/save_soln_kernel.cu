#include "user_defined_types.h"
#include "op_datatypes.h"
#include "kernels.h"
__device__
#include <save_soln.h>
__global__

void op_cuda_save_soln(float *arg0,float *arg1,int set_size)
{
  for (int n = threadIdx.x + blockIdx.x * blockDim.x; n < set_size; n += blockDim.x * gridDim.x) {
    save_soln(arg0 + n * 4,arg1 + n * 4);
  }
}


float op_par_loop_save_soln(const char *name,op_set set,struct op_dat<void> *arg0,int idx0,op_ptr *ptr0,enum op_access acc0,struct op_dat<void> *arg1,int idx1,op_ptr *ptr1,enum op_access acc1)
{
  int bsize = BSIZE;
  int gridsize = (set.size - 1) / bsize + 1;
  int reduct_bytes = 0;
  int reduct_size = 0;
  int reduct_shared = reduct_size * (BSIZE / 2);
  int const_bytes = 0;
cudaEvent_t start, stop;
  float elapsed_time_ms = 0.00000F;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  op_cuda_save_soln<<<gridsize,bsize,reduct_shared>>>(((float *)arg0->dat_d),((float *)arg1->dat_d),set.size);
  cudaEventRecord(stop,0);
  cudaThreadSynchronize();
  cudaEventElapsedTime(&elapsed_time_ms,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_time_ms;
}

