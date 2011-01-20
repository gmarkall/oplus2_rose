#include "user_defined_types.h"
#include "op_datatypes.h"
#include "kernels.h"
__device__
#include <res_calc.h>
__global__

void op_cuda_res_calc(float *ind_arg0,int *ind_arg0_ptrs,int *ind_arg0_sizes,int *ind_arg0_offset,float *ind_arg1,int *ind_arg1_ptrs,int *ind_arg1_sizes,int *ind_arg1_offset,float *ind_arg2,int *ind_arg2_ptrs,int *ind_arg2_sizes,int *ind_arg2_offset,float *ind_arg3,int *ind_arg3_ptrs,int *ind_arg3_sizes,int *ind_arg3_offset,int *arg0_ptrs,int *arg1_ptrs,int *arg2_ptrs,int *arg3_ptrs,int *arg4_ptrs,int *arg5_ptrs,int *arg6_ptrs,int *arg7_ptrs,int *arg8_d,int block_offset,int *blkmap,int *offset,int *nelems,int *ncolors,int *colors)
{
  float arg6_l[4];
  float arg7_l[4];
  int arg8_l[1];
  extern __shared__ 
  char shared[];
  __shared__ 
  int *ind_arg0_ptr;
  __shared__ 
  int *ind_arg1_ptr;
  __shared__ 
  int *ind_arg2_ptr;
  __shared__ 
  int *ind_arg3_ptr;
  __shared__ 
  int ind_arg0_size;
  __shared__ 
  int ind_arg1_size;
  __shared__ 
  int ind_arg2_size;
  __shared__ 
  int ind_arg3_size;
  __shared__ 
  float *ind_arg0_s;
  __shared__ 
  float *ind_arg1_s;
  __shared__ 
  float *ind_arg2_s;
  __shared__ 
  float *ind_arg3_s;
  __shared__ 
  int *arg0_ptr;
  __shared__ 
  int *arg1_ptr;
  __shared__ 
  int *arg2_ptr;
  __shared__ 
  int *arg3_ptr;
  __shared__ 
  int *arg4_ptr;
  __shared__ 
  int *arg5_ptr;
  __shared__ 
  int *arg6_ptr;
  __shared__ 
  int *arg7_ptr;
  __shared__ 
  int *arg8;
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
    ind_arg2_size = ind_arg2_sizes[blockId];
    ind_arg3_size = ind_arg3_sizes[blockId];
    ind_arg0_ptr = ind_arg0_ptrs + ind_arg0_offset[blockId];
    ind_arg1_ptr = ind_arg1_ptrs + ind_arg1_offset[blockId];
    ind_arg2_ptr = ind_arg2_ptrs + ind_arg2_offset[blockId];
    ind_arg3_ptr = ind_arg3_ptrs + ind_arg3_offset[blockId];
    arg0_ptr = arg0_ptrs + cur_offset;
    arg1_ptr = arg1_ptrs + cur_offset;
    arg2_ptr = arg2_ptrs + cur_offset;
    arg3_ptr = arg3_ptrs + cur_offset;
    arg4_ptr = arg4_ptrs + cur_offset;
    arg5_ptr = arg5_ptrs + cur_offset;
    arg6_ptr = arg6_ptrs + cur_offset;
    arg7_ptr = arg7_ptrs + cur_offset;
    arg8 = arg8_d + cur_offset * 1;
    int nbytes = 0;
    ind_arg0_s = ((float *)(&shared[nbytes]));
    nbytes += ROUND_UP(ind_arg0_size * (sizeof(float ) * 2));
    ind_arg1_s = ((float *)(&shared[nbytes]));
    nbytes += ROUND_UP(ind_arg1_size * (sizeof(float ) * 4));
    ind_arg2_s = ((float *)(&shared[nbytes]));
    nbytes += ROUND_UP(ind_arg2_size * (sizeof(float ) * 1));
    ind_arg3_s = ((float *)(&shared[nbytes]));
  }
  __syncthreads();
  for (int n = threadIdx.x; n < ind_arg0_size; n += blockDim.x) {
    int ind_index = ind_arg0_ptr[n];
    ind_arg0_s[0+n*2] = ind_arg0[0+ind_index*2];
    ind_arg0_s[1+n*2] = ind_arg0[1+ind_index*2];
  }
  for (int n = threadIdx.x; n < ind_arg1_size; n += blockDim.x) {
    int ind_index = ind_arg1_ptr[n];
    ind_arg1_s[0+n*4] = ind_arg1[0+ind_index*4];
    ind_arg1_s[1+n*4] = ind_arg1[1+ind_index*4];
    ind_arg1_s[2+n*4] = ind_arg1[2+ind_index*4];
    ind_arg1_s[3+n*4] = ind_arg1[3+ind_index*4];
  }
  for (int n = threadIdx.x; n < ind_arg2_size; n += blockDim.x) {
    ind_arg2_s[n*1] = ind_arg2[ind_arg2_ptr[n]*1];
  }
  for (int n = threadIdx.x; n < ind_arg3_size; n += blockDim.x) {
    ind_arg3_s[0+n*4] = 0;
    ind_arg3_s[1+n*4] = 0;
    ind_arg3_s[2+n*4] = 0;
    ind_arg3_s[3+n*4] = 0;
  }
  __syncthreads();
  for (int n = threadIdx.x; n < nelem2; n += blockDim.x) {
    int col2 = -1;
    if (n < nelem) {
      arg6_l[0] = 0;
      arg6_l[1] = 0;
      arg6_l[2] = 0;
      arg6_l[3] = 0;
      arg7_l[0] = 0;
      arg7_l[1] = 0;
      arg7_l[2] = 0;
      arg7_l[3] = 0;
      arg8_l[0] =  *(arg8 + (n * 1 + 0));
      res_calc(ind_arg0_s + arg0_ptr[n] * 2,ind_arg0_s + arg1_ptr[n] * 2,ind_arg1_s + arg2_ptr[n] * 4,ind_arg1_s + arg3_ptr[n] * 4,ind_arg2_s + arg4_ptr[n] * 1,ind_arg2_s + arg5_ptr[n] * 1,arg6_l,arg7_l,arg8_l);
      col2 = color[n];
    }
    for (int col = 0; col < ncolor; ++col) {
      if (col == col2) {
        int ind_index = arg6_ptr[n];
        ind_arg3_s[0+ind_index*4] += arg6_l[0];
        ind_arg3_s[1+ind_index*4] += arg6_l[1];
        ind_arg3_s[2+ind_index*4] += arg6_l[2];
        ind_arg3_s[3+ind_index*4] += arg6_l[3];
        ind_index = arg7_ptr[n];
        ind_arg3_s[0+ind_index*4] += arg7_l[0];
        ind_arg3_s[1+ind_index*4] += arg7_l[1];
        ind_arg3_s[2+ind_index*4] += arg7_l[2];
        ind_arg3_s[3+ind_index*4] += arg7_l[3];
      }
      __syncthreads();
    }
  }
  for (int n = threadIdx.x; n < ind_arg3_size; n += blockDim.x) {
    int ind_index = ind_arg3_ptr[n];
    ind_arg3[0+ind_index*4] += ind_arg3_s[0+n*4];
    ind_arg3[1+ind_index*4] += ind_arg3_s[1+n*4];
    ind_arg3[2+ind_index*4] += ind_arg3_s[2+n*4];
    ind_arg3[3+ind_index*4] += ind_arg3_s[3+n*4];
  }
}


float op_par_loop_res_calc(const char *name,op_set set,struct op_dat<void> *arg0,int idx0,op_ptr *ptr0,enum op_access acc0,struct op_dat<void> *arg1,int idx1,op_ptr *ptr1,enum op_access acc1,struct op_dat<void> *arg2,int idx2,op_ptr *ptr2,enum op_access acc2,struct op_dat<void> *arg3,int idx3,op_ptr *ptr3,enum op_access acc3,struct op_dat<void> *arg4,int idx4,op_ptr *ptr4,enum op_access acc4,struct op_dat<void> *arg5,int idx5,op_ptr *ptr5,enum op_access acc5,struct op_dat<void> *arg6,int idx6,op_ptr *ptr6,enum op_access acc6,struct op_dat<void> *arg7,int idx7,op_ptr *ptr7,enum op_access acc7,struct op_dat<void> *arg8,int idx8,op_ptr *ptr8,enum op_access acc8)
{
  int nargs = 9;
  int ninds = 4;
  int gridsize = (set.size - 1) / BSIZE + 1;
  struct op_dat<void> args[9] = { *arg0,  *arg1,  *arg2,  *arg3,  *arg4,  *arg5,  *arg6,  *arg7,  *arg8};
  int idxs[9] = {idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, -1};
  op_ptr ptrs[9] = { *ptr0,  *ptr1,  *ptr2,  *ptr3,  *ptr4,  *ptr5,  *ptr6,  *ptr7, OP_ID};
  int dims[9] = {arg0->dim, arg1->dim, arg2->dim, arg3->dim, arg4->dim, arg5->dim, arg6->dim, arg7->dim, arg8->dim};
  enum op_access accs[9] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8};
  int inds[9] = {0, 0, 1, 1, 2, 2, 3, 3, -1};
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
    op_cuda_res_calc<<<nblocks,BSIZE,nshared>>>(((float *)arg0->dat_d),Plan->ind_ptrs[0],Plan->ind_sizes[0],Plan->ind_offs[0],((float *)arg2->dat_d),Plan->ind_ptrs[1],Plan->ind_sizes[1],Plan->ind_offs[1],((float *)arg4->dat_d),Plan->ind_ptrs[2],Plan->ind_sizes[2],Plan->ind_offs[2],((float *)arg6->dat_d),Plan->ind_ptrs[3],Plan->ind_sizes[3],Plan->ind_offs[3],Plan->ptrs[0],Plan->ptrs[1],Plan->ptrs[2],Plan->ptrs[3],Plan->ptrs[4],Plan->ptrs[5],Plan->ptrs[6],Plan->ptrs[7],((int *)arg8->dat_d),block_offset,Plan->blkmap,Plan->offset,Plan->nelems,Plan->nthrcol,Plan->thrcol);
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

