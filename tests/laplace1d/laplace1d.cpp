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

//
// OP header file
//

#include "op_seq.h"


//
// kernel routines for parallel loops
//

#include "laplace.h"


// define problem size

#define NN       6
#define NITER    2


// main program

int main(int argc, char **argv){

  int   nnode;

  nnode = (NN+1);

  int    *p_elem_node = (int *)malloc(2*sizeof(int)*NN);
  float  *p_xn = (float *)malloc(sizeof(float)*2*nnode);
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
  op_dat<float> xn(nodes, 2, p_xn);

  op_sparsity mat_sparsity(elements, elem_node, elem_node);

  op_sparse_matrix<float> mat(mat_sparsity);

  op_diagnostic_output();

  // construct the matrix
  op_par_loop(laplace, elements, mat,
              elem_node, 0, elem_node, 0, OP_INC,
              &xn, 0, &elem_node, OP_READ);

  // spmv
  op_mat_mult(mat, x, y);

  op_fetch_data(&y);

}

