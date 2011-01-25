/*
  Open source copyright declaration based on BSD open source template:
  http://www.opensource.org/licenses/bsd-license.php
* Copyright (c) 2009, Mike Giles
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * The name of Mike Giles may not be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//
//     Nonlinear airfoil lift calculation
//
//     Written by Mike Giles, 2010, based on FORTRAN code
//     by Devendra Ghate and Mike Giles, 2005
//
//
// standard headers
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "kernels.h" 
using namespace std;
// global constants
float gam;
float gm1;
float cfl;
float eps;
float mach;
float alpha;
#include "user_defined_types.h"
myconst air_const = {(2.0f), (1.0f)};
//
// OP header file
//
#include "op_seq.h"
//
// kernel routines for parallel loops
//
#include "input.h"
#include "save_soln.h"
#include "adt_calc.h"
#include "res_calc.h"
#include "update.h"
class std::vector< int  , std::allocator< int  >  > *prev_parts = (0L);

int comp2(const void *a2,const void *b2)
{
  struct point *a = (struct point *)a2;
  struct point *b = (struct point *)b2;
  if (prev_parts ->  at (((a -> point::part))) == prev_parts ->  at (((b -> point::part)))) 
    return 0;
  else if (prev_parts ->  at (((a -> point::part))) < prev_parts ->  at (((b -> point::part)))) 
    return (-1);
  else 
    return 1;
}

// main program

int main(int argc,char **argv)
{
#define maxnode  9900
#define maxcell (9702+1)
#define maxedge 19502
  int ecell[39004UL];
  int boun[19502UL];
  int edge[39004UL];
  int cell[38812UL];
  float x[19800UL];
  float q[38812UL];
  float qold[38812UL];
  float adt[9703UL];
  float res[38812UL];
  int nnode;
  int ncell;
  int nedge;
  int niter;
  float rms;
// read in grid and flow data
  input(9900,(9702 + 1),19502,nnode,ncell,nedge,x,q,cell,edge,ecell,boun);
// obtain partition info
  int npart;
  struct point partnode[9900UL];
  struct point partcell[9703UL];
  struct point partedge[19502UL];
  intput_partition_info(((argv[1])),nnode,ncell,npart,partnode,partcell);
// create partition info
  class std::vector< int  , std::allocator< int  >  > node_part_info(npart,0);
  class std::vector< int  , std::allocator< int  >  > cell_part_info(npart,0);
  class std::vector< int  , std::allocator< int  >  > edge_part_info(npart,0);
// Cell based edge partition
  for (int i = 0; i < nedge; i++) {
    int part1 = (partcell[ecell[i * 2]].point::part);
    int part2 = (partcell[ecell[(i * 2) + 1]].point::part);
    int target_part = part2;
    partedge[i].point::part = target_part;
    partedge[i].point::index = i;
  }
// 1.1 fixup backtrack info - N/A
// 1.2 sort partedge
  int counter = 0;
  prev_parts = (::new std::vector< int  , std::allocator< int  >  > (npart,(-1)));
  for (int i = 0; i < nedge; i++) {
    int part = (partedge[i].point::part);
    if (prev_parts ->  at ((part)) < 0) 
      prev_parts ->  at ((part)) = counter++;
  }
  qsort((partedge),(nedge),((sizeof(struct point ))),comp2);
  delete prev_parts;
// 1.3 fixup edge_part_info
  counter = (-1);
  int old_part = (-1);
  for (int i = 0; i < nedge; i++) {
    int part = (partedge[i].point::part);
    if (old_part != part) {
      counter++;
    }
    old_part = part;
    edge_part_info[(counter)]++;
  }
// 1.4 fixup edge watchers i.e. N/A
// 1.5 sort actual edge pointers
  int *edge2 = new int [39004UL];
  int *ecell2 = new int [39004UL];
  int *boun2 = new int [19502UL];
  for (int i = 0; i < nedge; i++) {
    int index = (partedge[i].point::index);
    for (int j = 0; j < 2; j++) {
      edge2[(i * 2) + j] = (edge[(index * 2) + j]);
      ecell2[(i * 2) + j] = (ecell[(index * 2) + j]);
    }
    boun2[i] = (boun[index]);
  }
  memcpy((boun),(boun2),(((sizeof(int )) * (19502))));
  memcpy((edge),(edge2),((((sizeof(int )) * (2)) * (19502))));
  memcpy((ecell),(ecell2),((((sizeof(int )) * (2)) * (19502))));
  delete []edge2;
  delete []ecell2;
  delete []boun2;
// initialise residual
  for (int n = 0; n < (4 * ncell); n++) 
    res[n] = (0.0);
// OP initialisation
  op_init(argc,argv);
// declare sets, pointers, datasets and global constants
  op_set nodes((nnode),((0L)),"nodes");
  op_set edges((nedge),&edge_part_info,"edges");
  op_set cells((ncell),((0L)),"cells");
  op_ptr pedge((edges),(nodes),2,edge,"pedge");
  op_ptr pecell((edges),(cells),2,ecell,"pecell2");
  op_ptr pcell((cells),(nodes),4,cell,"pcell");
  struct op_dat< int  > p_boun((edges),1,(boun),"pp_boun");
  struct op_dat< float  > p_x((nodes),2,(x),"p_x");
  struct op_dat< float  > p_q((cells),4,(q),"p_q");
  struct op_dat< float  > p_qold((cells),4,(qold),"p_qold");
  struct op_dat< float  > p_adt((cells),1,(adt),"p_adt");
  struct op_dat< float  > p_res((cells),4,(res),"p_res");
  op_decl_const(1,&gam,"gam");
  op_decl_const(1,&gm1,"gm1");
  op_decl_const(1,&cfl,"cfl");
  op_decl_const(1,&eps,"eps");
  op_decl_const(1,&mach,"mach");
  op_decl_const(1,&alpha,"alpha");
  op_decl_const(1,&air_const,"air_const");
  op_diagnostic_output();
// main time-marching loop
  niter = 10;
  float saveTime = 0.0f;
  float adtTime = 0.0f;
  float resTime = 0.0f;
  float updateTime = 0.0f;
  for (int iter = 1; iter <= niter; iter++) {
//  save old flow solution
    saveTime += op_par_loop_save_soln("save_soln",(cells),(struct op_dat<void> *)(&p_q),0,((0L)),OP_READ,(struct op_dat<void> *)(&p_qold),0,((0L)),OP_WRITE);
//  predictor/corrector update loop
    for (int k = 0; k < 2; k++) {
//    calculate area/timstep
      adtTime += op_par_loop_adt_calc("adt_calc",(cells),(struct op_dat<void> *)(&p_x),0,&pcell,OP_READ,(struct op_dat<void> *)(&p_x),1,&pcell,OP_READ,(struct op_dat<void> *)(&p_x),2,&pcell,OP_READ,(struct op_dat<void> *)(&p_x),3,&pcell,OP_READ,(struct op_dat<void> *)(&p_q),0,((0L)),OP_READ,(struct op_dat<void> *)(&p_adt),0,((0L)),OP_WRITE);
//    calculate flux residual
      resTime += op_par_loop_res_calc("res_calc",(edges),(struct op_dat<void> *)(&p_x),0,&pedge,OP_READ,(struct op_dat<void> *)(&p_x),1,&pedge,OP_READ,(struct op_dat<void> *)(&p_q),0,&pecell,OP_READ,(struct op_dat<void> *)(&p_q),1,&pecell,OP_READ,(struct op_dat<void> *)(&p_adt),0,&pecell,OP_READ,(struct op_dat<void> *)(&p_adt),1,&pecell,OP_READ,(struct op_dat<void> *)(&p_res),0,&pecell,OP_INC,(struct op_dat<void> *)(&p_res),1,&pecell,OP_INC,(struct op_dat<void> *)(&p_boun),0,((0L)),OP_READ);
//    update flow field
      rms = (0.0);
      struct op_dat_gbl< float  > p_rms(1,((&rms)),"p_rms");
      updateTime += op_par_loop_update("update",(cells),(struct op_dat<void> *)(&p_qold),0,((0L)),OP_READ,(struct op_dat<void> *)(&p_q),0,((0L)),OP_WRITE,(struct op_dat<void> *)(&p_res),0,((0L)),OP_RW,(struct op_dat<void> *)(&p_adt),0,((0L)),OP_READ,(struct op_dat<void> *)((&p_rms)),0,((0L)),OP_INC);
    }
//  print iteration history
    rms = ((sqrt(((rms / ((float )ncell))))));
    printf(" %d  %10.5e \n",iter,(rms));
  }
  return 0;
}

