#include "user_defined_types.h"
#include "op_datatypes.h"
#include "kernels.h"
__device__
#include <adt_calc.h>
__global__

void op_cuda_adt_calc(float *ind_arg0,int *ind_arg0_maps,int *ind_arg0_sizes,int *ind_arg0_offset,short *arg0_maps,short *arg1_maps,short *arg2_maps,short *arg3_maps,float *arg4_d,float *arg5_d,int block_offset,int *blkmap,int *offset,int *nelems,int *ncolors,int *colors)
{
  float arg4_l[4];
  float arg5_l[1];
  extern __shared__ 
  char shared[];
  __shared__ 
  int *ind_arg0_map;
  __shared__ 
  int ind_arg0_size;
  __shared__ 
  float *ind_arg0_s;
  __shared__ 
  short *arg0_map;
  __shared__ 
  short *arg1_map;
  __shared__ 
  short *arg2_map;
  __shared__ 
  short *arg3_map;
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
    short cur_offset = offset[blockId];
    color = colors + cur_offset;
    nelem2 = blockDim.x * (1 + (nelem - 1) / blockDim.x);
    ind_arg0_size = ind_arg0_sizes[blockId];
    ind_arg0_map = ind_arg0_maps + ind_arg0_offset[blockId];
    arg0_map = arg0_maps + cur_offset;
    arg1_map = arg1_maps + cur_offset;
    arg2_map = arg2_maps + cur_offset;
    arg3_map = arg3_maps + cur_offset;
    arg4 = arg4_d + cur_offset * 4;
    arg5 = arg5_d + cur_offset * 1;
    int nbytes = 0;
    ind_arg0_s = ((float *)(&shared[nbytes]));
  }
  __syncthreads();
  for (int n = threadIdx.x; n < ind_arg0_size; n += blockDim.x) {
    int ind_index = ind_arg0_map[n];
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
      adt_calc(ind_arg0_s + arg0_map[n] * 2,ind_arg0_s + arg1_map[n] * 2,ind_arg0_s + arg2_map[n] * 2,ind_arg0_s + arg3_map[n] * 2,arg4_l,arg5_l);
       *(arg5 + (n * 1 + 0)) = arg5_l[0];
      col2 = color[n];
    }
  }
}


float op_par_loop_adt_calc(const char *name,op_set set,struct op_dat<void> *arg0,int idx0,op_map *map0,enum op_access acc0,struct op_dat<void> *arg1,int idx1,op_map *map1,enum op_access acc1,struct op_dat<void> *arg2,int idx2,op_map *map2,enum op_access acc2,struct op_dat<void> *arg3,int idx3,op_map *map3,enum op_access acc3,struct op_dat<void> *arg4,int idx4,op_map *map4,enum op_access acc4,struct op_dat<void> *arg5,int idx5,op_map *map5,enum op_access acc5)
{
  int nargs = 6;
  int ninds = 1;
  int gridsize = (set.size - 1) / OP_block_size + 1;
  struct op_dat<void> args[6] = { *arg0,  *arg1,  *arg2,  *arg3,  *arg4,  *arg5};
  int idxs[6] = {idx0, idx1, idx2, idx3, -1, -1};
  op_map maps[6] = { *map0,  *map1,  *map2,  *map3, OP_ID, OP_ID};
  int dims[6] = {arg0->dim, arg1->dim, arg2->dim, arg3->dim, arg4->dim, arg5->dim};
  enum op_access accs[6] = {acc0, acc1, acc2, acc3, acc4, acc5};
  int inds[6] = {0, 0, 0, 0, -1, -1};
  op_plan *Plan = plan(name,set,nargs,args,idxs,maps,dims,accs,ninds,inds);
  int block_offset = 0;
  int reduct_bytes = 0;
  int reduct_size = 0;
  int reduct_shared = reduct_size * (OP_block_size / 2);
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
    op_cuda_adt_calc<<<nblocks,OP_block_size,nshared>>>(((float *)arg0->dat_d),Plan->ind_maps[0],Plan->ind_sizes,Plan->ind_offs,Plan->maps[0],Plan->maps[1],Plan->maps[2],Plan->maps[3],((float *)arg4->dat_d),((float *)arg5->dat_d),block_offset,Plan->blkmap,Plan->offset,Plan->nelems,Plan->nthrcol,Plan->thrcol);
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

