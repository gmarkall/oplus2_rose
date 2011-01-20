#include "user_defined_types.h"
#include "op_datatypes.h"
#include "kernels.h"
__device__
#include <adt_calc.h>
__global__

void op_cuda_adt_calc(float *ind_arg0,int *ind_arg0_ptrs,int *ind_arg0_sizes,int *ind_arg0_offset,int *arg0_ptrs,int *arg1_ptrs,int *arg2_ptrs,int *arg3_ptrs,float *arg4_d,float *arg5_d,int block_offset,int *blkmap,int *offset,int *nelems,int *ncolors,int *colors)
{
  float arg4_l[4];
  float arg5_l[1];
  extern __shared__ 
  char shared[];
  __shared__ 
  int *ind_arg0_ptr;
  __shared__ 
  int ind_arg0_size;
  __shared__ 
  float *ind_arg0_s;
  __shared__ 
  int *arg0_ptr;
  __shared__ 
  int *arg1_ptr;
  __shared__ 
  int *arg2_ptr;
  __shared__ 
  int *arg3_ptr;
  __shared__ 
  float *arg4;
  __shared__ 
  float *arg5;
  __shared__ 
  int nelem2;
  __shared__ 
  int ncolor;
  __shared__ 
  int *color;
  __shared__ 
  int blockId;
  __shared__ 
  int nelem;
  if (threadIdx.x == 0) {
    blockId = blkmap[blockIdx.x + block_offset];
    nelem = nelems[blockId];
    ncolor = ncolors[blockId];
    int cur_offset = offset[blockId];
    color = colors + cur_offset;
    nelem2 = blockDim.x * (1 + (nelem - 1) / blockDim.x);
    ind_arg0_size = ind_arg0_sizes[blockId];
    ind_arg0_ptr = ind_arg0_ptrs + ind_arg0_offset[blockId];
    arg0_ptr = arg0_ptrs + cur_offset;
    arg1_ptr = arg1_ptrs + cur_offset;
    arg2_ptr = arg2_ptrs + cur_offset;
    arg3_ptr = arg3_ptrs + cur_offset;
    arg4 = arg4_d + cur_offset * 4;
    arg5 = arg5_d + cur_offset * 1;
    int nbytes = 0;
    ind_arg0_s = ((float *)(&shared[nbytes]));
  }
  __syncthreads();
  for (int n = threadIdx.x; n < ind_arg0_size; n += blockDim.x) {
    int ind_index = ind_arg0_ptr[n];
    ind_arg0_s[0+n*2] = ind_arg0[0+ind_index*2];
    ind_arg0_s[1+n*2] = ind_arg0[1+ind_index*2];
  }
  __syncthreads();
  for (int n = threadIdx.x; n < nelem2; n += blockDim.x) {
    int col2 = -1;
    if (n < nelem) {
      arg4_l[0] =  *(arg4 + (n * 4 + 0));
      arg4_l[1] =  *(arg4 + (n * 4 + 1));
      arg4_l[2] =  *(arg4 + (n * 4 + 2));
      arg4_l[3] =  *(arg4 + (n * 4 + 3));
      adt_calc(ind_arg0_s + arg0_ptr[n] * 2,ind_arg0_s + arg1_ptr[n] * 2,ind_arg0_s + arg2_ptr[n] * 2,ind_arg0_s + arg3_ptr[n] * 2,arg4_l,arg5_l);
       *(arg5 + (n * 1 + 0)) = arg5_l[0];
      col2 = color[n];
    }
  }
}


float op_par_loop_adt_calc(const char *name,op_set set,struct op_dat<void> *arg0,int idx0,op_ptr *ptr0,enum op_access acc0,struct op_dat<void> *arg1,int idx1,op_ptr *ptr1,enum op_access acc1,struct op_dat<void> *arg2,int idx2,op_ptr *ptr2,enum op_access acc2,struct op_dat<void> *arg3,int idx3,op_ptr *ptr3,enum op_access acc3,struct op_dat<void> *arg4,int idx4,op_ptr *ptr4,enum op_access acc4,struct op_dat<void> *arg5,int idx5,op_ptr *ptr5,enum op_access acc5)
{
  int nargs = 6;
  int ninds = 1;
  int gridsize = (set.size - 1) / BSIZE + 1;
  struct op_dat<void> args[6] = { *arg0,  *arg1,  *arg2,  *arg3,  *arg4,  *arg5};
  int idxs[6] = {idx0, idx1, idx2, idx3, -1, -1};
  op_ptr ptrs[6] = { *ptr0,  *ptr1,  *ptr2,  *ptr3, OP_ID, OP_ID};
  int dims[6] = {arg0->dim, arg1->dim, arg2->dim, arg3->dim, arg4->dim, arg5->dim};
  enum op_access accs[6] = {acc0, acc1, acc2, acc3, acc4, acc5};
  int inds[6] = {0, 0, 0, 0, -1, -1};
  op_plan *Plan = plan(name,set,nargs,args,idxs,ptrs,dims,accs,ninds,inds);
  int block_offset = 0;
  int reduct_bytes = 0;
  int reduct_size = 0;
  int reduct_shared = reduct_size * (BSIZE / 2);
  int const_bytes = 0;
  float total_time = 0.00000F;
  for (int col = 0; col < Plan->ncolors; ++col) {
    int nblocks = Plan->ncolblk[col];
    int nshared = Plan->nshared;
cudaEvent_t start, stop;
    float elapsed_time_ms = 0.00000F;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    op_cuda_adt_calc<<<nblocks,BSIZE,nshared>>>(((float *)arg0->dat_d),Plan->ind_ptrs[0],Plan->ind_sizes[0],Plan->ind_offs[0],Plan->ptrs[0],Plan->ptrs[1],Plan->ptrs[2],Plan->ptrs[3],((float *)arg4->dat_d),((float *)arg5->dat_d),block_offset,Plan->blkmap,Plan->offset,Plan->nelems,Plan->nthrcol,Plan->thrcol);
    cudaEventRecord(stop,0);
    cudaThreadSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    total_time += elapsed_time_ms;
    cudaThreadSynchronize();
    block_offset += nblocks;
  }
  return total_time;
}

