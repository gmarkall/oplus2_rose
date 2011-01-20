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

float alpha[2];

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

int main(int argc, char **argv){

  int   nnode, nedge, n, e;
  float dx;

  nnode = (NN-1)*(NN-1);
  nedge = (NN-1)*(NN-1) + 4*(NN-1)*(NN-2);
  dx    = 1.0f / ((float) NN);

  int    *p1 = (int *)malloc(sizeof(int)*nedge);
  int    *p2 = (int *)malloc(sizeof(int)*nedge);

  float  *xe = (float *)malloc(sizeof(float)*2*nedge);
  float  *xn = (float *)malloc(sizeof(float)*2*nnode);

  double *A  = (double *)malloc(sizeof(double)*nedge);
  float  *r  = (float *)malloc(sizeof(float)*nnode);
  float  *u  = (float *)malloc(sizeof(float)*nnode);
  float  *du = (float *)malloc(sizeof(float)*nnode);

  // create matrix and r.h.s., and set coordinates needed for renumbering / partitioning

  e = 0;

  for (int i=1; i<NN; i++) {
    for (int j=1; j<NN; j++) {
      n         = i-1 + (j-1)*(NN-1);
      r[n]      = 0.0f;
      u[n]      = 0.0f;
      du[n]     = 0.0f;
      xn[2*n  ] = i*dx;
      xn[2*n+1] = j*dx;

      p1[e]     = n;
      p2[e]     = n;
      A[e]      = -1.0f;
      xe[2*e  ] = i*dx;
      xe[2*e+1] = j*dx;
      e++;

      for (int pass=0; pass<4; pass++) {
        int i2 = i;
        int j2 = j;
        if (pass==0) i2 += -1;
        if (pass==1) i2 +=  1;
        if (pass==2) j2 += -1;
        if (pass==3) j2 +=  1;

        if ( (i2==0) || (i2==NN) || (j2==0) || (j2==NN) ) {
          r[n] += 0.25f;
	}
        else {
          p1[e]     = n;
          p2[e]     = i2-1 + (j2-1)*(NN-1);
          A[e]      = 0.25f;
          xe[2*e  ] = i*dx;
          xe[2*e+1] = j*dx;
          e++;
        }
      }
    }
  }

  // OP initialisation

  op_init(argc,argv);

  // declare sets, pointers, and datasets

  op_set nodes(nnode, NULL);
  op_set edges(nedge, NULL);

  op_ptr pedge1(edges,nodes,1,p1);
  op_ptr pedge2(edges,nodes,1,p2);

  op_dat<double> p_A(edges,1, A);
  op_dat<float> p_r(nodes,1, r);
  op_dat<float> p_u(nodes,1, u);
  op_dat<float> p_du(nodes,1, du);

  alpha[0] = 1.0f;
  alpha[1] = 1.0f;
  op_decl_const(2,alpha);

  op_diagnostic_output();

  // main iteration loop

  float u_sum, u_max, beta = 1.0f;
  op_dat_gbl<float> p_u_sum(1,&u_sum);
  op_dat_gbl<float> p_u_max(1,&u_max);
  op_dat_gbl<float> p_beta(1,&beta);

  for (int iter=0; iter<NITER; iter++) {
    op_par_loop_4(res,edges,
                  &p_A,    0,NULL,    OP_READ,
                  &p_u,    0,&pedge2, OP_READ,
                  &p_du,   0,&pedge1, OP_INC,
		  &p_beta, 0,NULL,    OP_READ);

    u_sum = 0.0f;
    u_max = 0.0f;
    op_par_loop_5(update,nodes,
                  &p_r,    0,NULL, OP_READ,
                  &p_du,   0,NULL, OP_RW,
                  &p_u,    0,NULL, OP_INC,
		  &p_u_sum,0,NULL, OP_INC,
                  &p_u_max,0,NULL, OP_MAX);
    printf("\n u max/rms = %f %f \n\n",u_max, sqrt(u_sum/nnode));
  }

  // print out results

  printf("\n  Results after %d iterations:\n\n",NITER);

  op_fetch_data((op_dat<float> *)&p_u);
  /*
  op_fetch_data(p_du);
  op_fetch_data(p_r);
  */

  for (int pass=0; pass<1; pass++) {
    /*
    if(pass==0)      printf("\narray u\n");
    else if(pass==1) printf("\narray du\n");
    else if(pass==2) printf("\narray r\n");
    */

    for (int j=NN-1; j>0; j--) {
      for (int i=1; i<NN; i++) {
        if (pass==0)
        printf(" %7.4f",u[i-1 + (j-1)*(NN-1)]);
        else if (pass==1)
        printf(" %7.4f",du[i-1 + (j-1)*(NN-1)]);
        else if (pass==2)
        printf(" %7.4f",r[i-1 + (j-1)*(NN-1)]);
      }
      printf("\n");
    }
    printf("\n");
  }
}





