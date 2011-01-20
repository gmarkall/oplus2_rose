//
// test program for new OPlus2 development
//
//
// standard headers
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
// global constants
#include "kernels.h" 
float alpha[2UL];
/*
   0            none
   1 or above   error-checking
   2 or above   report additional info
*/
//
// OP header file
//
#include "op_seq.h"
//
// kernel routines for parallel loops
//
#include "res.h"
#include "update.h"
// define problem size
#define NN       6
#define NITER    2
// main program

int main(int argc,char **argv)
{
  int nnode;
  int nedge;
  int n;
  int e;
  float dx;
  nnode = ((6 - 1) * (6 - 1));
  nedge = (((6 - 1) * (6 - 1)) + ((4 * (6 - 1)) * (6 - 2)));
  dx = (1.0f / ((float )6));
  int *p1 = (int *)(malloc((((sizeof(int )) * (nedge)))));
  int *p2 = (int *)(malloc((((sizeof(int )) * (nedge)))));
  float *xe = (float *)(malloc(((((sizeof(float )) * (2)) * (nedge)))));
  float *xn = (float *)(malloc(((((sizeof(float )) * (2)) * (nnode)))));
  double *A = (double *)(malloc((((sizeof(double )) * (nedge)))));
  float *r = (float *)(malloc((((sizeof(float )) * (nnode)))));
  float *u = (float *)(malloc((((sizeof(float )) * (nnode)))));
  float *du = (float *)(malloc((((sizeof(float )) * (nnode)))));
// create matrix and r.h.s., and set coordinates needed for renumbering / partitioning
  e = 0;
  for (int i = 1; i < 6; i++) {
    for (int j = 1; j < 6; j++) {
      n = ((i - 1) + ((j - 1) * (6 - 1)));
      r[n] = 0.0f;
      u[n] = 0.0f;
      du[n] = 0.0f;
      xn[2 * n] = ((i) * dx);
      xn[(2 * n) + 1] = ((j) * dx);
      p1[e] = n;
      p2[e] = n;
      A[e] = ((-1.0f));
      xe[2 * e] = ((i) * dx);
      xe[(2 * e) + 1] = ((j) * dx);
      e++;
      for (int pass = 0; pass < 4; pass++) {
        int i2 = i;
        int j2 = j;
        if (pass == 0) 
          i2 += (-1);
        if (pass == 1) 
          i2 += 1;
        if (pass == 2) 
          j2 += (-1);
        if (pass == 3) 
          j2 += 1;
        if ((((i2 == 0) || (i2 == 6)) || (j2 == 0)) || (j2 == 6)) {
          r[n] += 0.25f;
        }
        else {
          p1[e] = n;
          p2[e] = ((i2 - 1) + ((j2 - 1) * (6 - 1)));
          A[e] = (0.25f);
          xe[2 * e] = ((i) * dx);
          xe[(2 * e) + 1] = ((j) * dx);
          e++;
        }
      }
    }
  }
// OP initialisation
  op_init(argc,argv);
// declare sets, pointers, and datasets
  op_set nodes((nnode),((0L)),"nodes");
  op_set edges((nedge),((0L)),"edges");
  op_ptr pedge1((edges),(nodes),1,p1,"pedge1");
  op_ptr pedge2((edges),(nodes),1,p2,"pedge2");
  struct op_dat< double  > p_A((edges),1,(A),"p_A");
  struct op_dat< float  > p_r((nodes),1,(r),"p_r");
  struct op_dat< float  > p_u((nodes),1,(u),"p_u");
  struct op_dat< float  > p_du((nodes),1,(du),"p_du");
  alpha[0] = 1.0f;
  alpha[1] = 1.0f;
  op_decl_const(2,alpha,"alpha");
  op_diagnostic_output();
// main iteration loop
  float u_sum;
  float u_max;
  float beta = 1.0f;
  struct op_dat_gbl< float  > p_u_sum(1,((&u_sum)),"p_u_sum");
  struct op_dat_gbl< float  > p_u_max(1,((&u_max)),"p_u_max");
  struct op_dat_gbl< float  > p_beta(1,((&beta)),"p_beta");
  for (int iter = 0; iter < 2; iter++) {
    op_par_loop_res("res",(edges),(struct op_dat<void> *)(&p_A),0,((0L)),OP_READ,(struct op_dat<void> *)(&p_u),0,&pedge2,OP_READ,(struct op_dat<void> *)(&p_du),0,&pedge1,OP_INC,(struct op_dat<void> *)((&p_beta)),0,((0L)),OP_READ);
    u_sum = 0.0f;
    u_max = 0.0f;
    op_par_loop_update("update",(nodes),(struct op_dat<void> *)(&p_r),0,((0L)),OP_READ,(struct op_dat<void> *)(&p_du),0,((0L)),OP_RW,(struct op_dat<void> *)(&p_u),0,((0L)),OP_INC,(struct op_dat<void> *)((&p_u_sum)),0,((0L)),OP_INC,(struct op_dat<void> *)((&p_u_max)),0,((0L)),OP_MAX);
    printf("\n u max/rms = %f %f \n\n",(u_max),sqrt(((u_sum / (nnode)))));
  }
// print out results
  printf("\n  Results after %d iterations:\n\n",2);
  op_fetch_data(&p_u);
/*
  op_fetch_data(p_du);
  op_fetch_data(p_r);
  */
  for (int pass = 0; pass < 1; pass++) {
/*
    if(pass==0)      printf("\narray u\n");
    else if(pass==1) printf("\narray du\n");
    else if(pass==2) printf("\narray r\n");
    */
    for (int j = (6 - 1); j > 0; j--) {
      for (int i = 1; i < 6; i++) {
        if (pass == 0) 
          printf(" %7.4f",((u[(i - 1) + ((j - 1) * (6 - 1))])));
        else if (pass == 1) 
          printf(" %7.4f",((du[(i - 1) + ((j - 1) * (6 - 1))])));
        else if (pass == 2) 
          printf(" %7.4f",((r[(i - 1) + ((j - 1) * (6 - 1))])));
      }
      printf("\n");
    }
    printf("\n");
  }
  return 0;
}

