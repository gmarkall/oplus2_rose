// test program demonstrating matrix-free spmv for FE
// discretisation of a 1D Laplace operator in OP2

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OP header file

#include "op_seq.h"

// kernel routines for parallel loops

#include "laplace.h"

// define problem size

#define NN       6
#define NITER    2

// main program

int main(int argc, char **argv){

  int   nnode = (NN+1);

  int    *p_elem_node = (int *)malloc(2*sizeof(int)*NN);
  float  *p_xn = (float *)malloc(sizeof(float)*nnode);
  float  *p_x  = (float *)malloc(sizeof(float)*nnode);
  float  *p_y  = (float *)malloc(sizeof(float)*nnode);

  // create element -> node mapping
  for (int i = 0; i < NN; ++i) {
    p_elem_node[2*i] = i;
    p_elem_node[2*i+1] = i+1;
  }

  // create coordinates and populate x with -1/pi^2*sin(pi*x)
  for (int i = 0; i < nnode; ++i) {
    p_xn[i] = sin(0.5*M_PI*i/NN);
    p_x[i] = -1./(M_PI*M_PI)*sin(M_PI*p_xn[i])
  }

  // OP initialisation

  op_init(argc,argv);

  // declare sets, pointers, and datasets

  op_set nodes(nnode, NULL);
  op_set elements(NN, NULL);

  op_map elem_node(elements,nodes,2,p_elem_node);

  op_dat<float> x(nodes, 1, p_x);
  op_dat<float> y(nodes, 1, p_y);
  op_dat<float> xn(nodes, 1, p_xn);

  // matrix-free spmv
  op_par_loop(laplace, elements,
              &y,  OP_COLON, &elem_node, OP_INC,
              &x,  OP_COLON, &elem_node, OP_READ,
              &xn, OP_COLON, &elem_node, OP_READ);

}

