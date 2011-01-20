#include "user_defined_types.h"
#include "op_datatypes.h"
#include "kernels.h"
__device__
#include <res.h>
__global__

void op_cuda_res(float *ind_arg0,int *ind_arg0_ptrs,int *ind_arg0_sizes,int *ind_arg0_offset,float *ind_arg1,int *ind_arg1_ptrs,int *ind_arg1_sizes,int *ind_arg1_offset,double *arg0_d,int *arg1_ptrs,int *arg2_ptrs,float *arg3,int block_offset,int *blkmap,int *offset,int *nelems,int *ncolors,int *colors)
{
  double arg0_l[1];
  float arg2_l[1];
  extern __shared__ 
  char shared[];
  __shared__ 
  int *ind_arg0_ptr;
  __shared__ 
  int *ind_arg1_ptr;
  __shared__ 
  int ind_arg0_size;
  __shared__ 
  int ind_arg1_size;
  __shared__ 
  float *ind_arg0_s;
  __shared__ 
  float *ind_arg1_s;
  __shared__ 
  double *arg0;
  __shared__ 
  int *arg1_ptr;
  __shared__ 
  int *arg2_ptr;
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
    ind_arg1_size = ind_arg1_sizes[blockId];
    ind_arg0_ptr = ind_arg0_ptrs + ind_arg0_offset[blockId];
    ind_arg1_ptr = ind_arg1_ptrs + ind_arg1_offset[blockId];
    arg0 = arg0_d + cur_offset * 1;
    arg1_ptr = arg1_ptrs + cur_offset;
    arg2_ptr = arg2_ptrs + cur_offset;
    int nbytes = 0;
    ind_arg0_s = ((float *)(&shared[nbytes]));
    nbytes += ROUND_UP(ind_arg0_size * (sizeof(float ) * 1));
    ind_arg1_s = ((float *)(&shared[nbytes]));
  }
  __syncthreads();
  for (int n = threadIdx.x; n < ind_arg0_size; n += blockDim.x) {
    ind_arg0_s[n*1] = ind_arg0[ind_arg0_ptr[n]*1];
  }
  for (int n = threadIdx.x; n < ind_arg1_size; n += blockDim.x) {
    ind_arg1_s[n*1] = 0;
  }
  __syncthreads();
  for (int n = threadIdx.x; n < nelem2; n += blockDim.x) {
    int col2 = -1;
    if (n < nelem) {
      arg2_l[0] = 0;
      arg0_l[0] =  *(arg0 + (n * 1 + 0));
      res(arg0_l,ind_arg0_s + arg1_ptr[n] * 1,arg2_l,arg3);
      col2 = color[n];
    }
    for (int col = 0; col < ncolor; ++col) {
      if (col == col2) {
        ind_arg1_s[arg2_ptr[n]*1] += arg2_l[0];
      }
      __syncthreads();
    }
  }
  for (int n = threadIdx.x; n < ind_arg1_size; n += blockDim.x) {
    ind_arg1[ind_arg1_ptr[n]*1] = ind_arg1_s[n*1];
  }
}


float op_par_loop_res(const char *name,op_set set,struct op_dat<void> *arg0,int idx0,op_ptr *ptr0,enum op_access acc0,struct op_dat<void> *arg1,int idx1,op_ptr *ptr1,enum op_access acc1,struct op_dat<void> *arg2,int idx2,op_ptr *ptr2,enum op_access acc2,struct op_dat<void> *arg3,int idx3,op_ptr *ptr3,enum op_access acc3)
{
  int nargs = 4;
  int ninds = 2;
  int gridsize = (set.size - 1) / BSIZE + 1;
  struct op_dat<void> args[4] = { *arg0,  *arg1,  *arg2,  *arg3};
  int idxs[4] = {-1, idx1, idx2, -1};
  op_ptr ptrs[4] = {OP_ID,  *ptr1,  *ptr2, OP_ID};
  int dims[4] = {arg0->dim, arg1->dim, arg2->dim, arg3->dim};
  enum op_access accs[4] = {acc0, acc1, acc2, acc3};
  int inds[4] = {-1, 0, 1, -1};
  op_plan *Plan = plan(name,set,nargs,args,idxs,ptrs,dims,accs,ninds,inds);
  int block_offset = 0;
  int reduct_bytes = 0;
  int reduct_size = 0;
  int reduct_shared = reduct_size * (BSIZE / 2);
  int const_bytes = 0;
  const_bytes += ROUND_UP(1 * sizeof(float ));
  reallocConstArrays(const_bytes);
  const_bytes = 0;
  push_op_dat_as_const(*arg3,const_bytes);
  const_bytes += ROUND_UP(1 * sizeof(float ));
  mvConstArraysToDevice(const_bytes);
  float total_time = 0.00000F;
  for (int col = 0; col < Plan->ncolors; ++col) {
    int nblocks = Plan->ncolblk[col];
    int nshared = Plan->nshared;
cudaEvent_t start, stop;
    float elapsed_time_ms = 0.00000F;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    op_cuda_res<<<nblocks,BSIZE,nshared>>>(((float *)arg1->dat_d),Plan->ind_ptrs[0],Plan->ind_sizes[0],Plan->ind_offs[0],((float *)arg2->dat_d),Plan->ind_ptrs[1],Plan->ind_sizes[1],Plan->ind_offs[1],((double *)arg0->dat_d),Plan->ptrs[1],Plan->ptrs[2],((float *)arg3->dat_d),block_offset,Plan->blkmap,Plan->offset,Plan->nelems,Plan->nthrcol,Plan->thrcol);
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

