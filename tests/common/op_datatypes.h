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
// OP datatypes
//

#ifndef OP_DATATYPES
#define OP_DATATYPES

#include <vector>

//
// OP diagnostics level; defined in op_seq.cpp/op_lib.cu and set in op_init
//

extern int OP_diags;

/*
   0            none
   1 or above   error-checking
   2 or above   info on plan construction
   3 or above   report execution of parallel loops
   4 or above   report use of old plans
   7 or above   report positive checks in op_plan_check
*/

//
// enum list for op_par_loop
//

enum op_access   { OP_READ, OP_WRITE, OP_RW, OP_INC, OP_MIN, OP_MAX };

//
// add in user's datatypes
//

// OP_USER_DATATYPES needs to contain "" or <> as appropriate
#ifdef OP_USER_DATATYPES
#include OP_USER_DATATYPES
#endif

//
// zero constants
//

#define ZERO_double  0.0;
#define ZERO_float   0.0f;
#define ZERO_int     0;
#define ZERO_uint    0;
#define ZERO_ll      0;
#define ZERO_ull     0;
#define ZERO_bool    0;

// null set
#define OP_NULL_SET op_set()

// identity mapping
#define OP_ID op_map()

//
// structures
//

using namespace std;

// struct: op_set
typedef struct _op_set
{
  int         size,   											// number of elements in set
              index;  											// index into list of sets 
  char const *name;   											// name of set
  vector<int>* partinfo;										// holds partition info for the set

  _op_set();
  _op_set(unsigned int size, vector<int>* partinfo, char const *name = "");
  ~_op_set();

protected:
  bool is_null();
}op_set;

// struct: op_map
typedef struct _op_map 
{
  op_set      from,   // set pointed from
              to;     // set pointed to
  int         dim,    // dimension of pointer
              index,  // index into list of pointers
             *map;    // array defining pointer
  char const *name;   // name of pointer

  _op_map();
  _op_map(op_set from, op_set to, int dim, int *map, char const *name = "");
  ~_op_map();
}op_map;

// struct: op_dat
template <class T>
struct op_dat
{
  op_set      set;    // set on which data is defined
  int         dim,    // dimension of data
              index,  // index into list of datasets
              size;   // size of each element in dataset
  char       *dat,    // data on host
             *dat_d,  // data on device (GPU)
             *dat_t;  // temporary data pointer
  char const *type,   // datatype
             *name;   // name of dataset

public:
  op_dat(op_set set, int dim, void* dat, char const *name = ""){
		init_as_data(set, dim, (char*)dat, sizeof(T), name);
	}

  ~op_dat(){}

protected:
  op_dat(){}

  void init_as_data(op_set set, int dim, char *dat, size_t size, char const *name);
};

extern void fixup_op_dat_data(op_dat<void>* data);

template <class T>
void op_dat<T>::init_as_data(op_set set, int dim, char* dat, size_t size, char const* name)
{
  this->set   = set;
  this->dim   = dim;
  this->dat   = dat;
  this->name  = name;
  this->size  = dim*size;

  fixup_op_dat_data((op_dat<void> *)this);
}

// struct op_dat_gbl
template <class T>
struct op_dat_gbl : public op_dat<T>
{
public:
  op_dat_gbl(int dim, void* dat, char const *name = ""){
    init_as_gbl(dim, (T*)dat, name);
  }

protected:
  void init_as_gbl(int dim, T *dat, char const *name)
  {
    this->set   = OP_NULL_SET;
    this->dim   = dim;
    this->index = 0;
    this->size  = dim*sizeof(T);
    this->dat   = (char *) dat;
    this->dat_t = (char *) dat;
    this->dat_d = NULL;
    this->name  = name;
  }
};

// struct: plan
typedef struct _op_plan{
  // input arguments
  char const  *name;
  int          set_index, nargs;
  int         *arg_idxs, *idxs, *map_idxs, *dims;
  op_access   *accs;

  // execution plan
  int        *nthrcol;  // number of thread colors for each block
  int        *thrcol;   // thread colors for each element of the primary set
  int        *offset;   // offset for primary set for the beginning of each block
  int       **ind_maps; // pointers for indirect datasets: 2D array, the outer
                        // index over the indirect datasets, and the inner one
                        // giving the local → global renumbering for the
                        // elements of the indirect set
  int        *ind_offs; // starting offsets for into ind_maps for each block
                        // for each indirect dataset
  int        *ind_sizes;// number of indirect indices for each block
                        // for each indirect dataset
  short     **maps;     // regular pointers, renumbered as needed
  int        *nelems;   // number of elements in each block
  int         ncolors;  // number of block colors
  int        *ncolblk;  // number of blocks for each color
  int        *blkmap;   // mapping to blocks of each color
  int         nshared;  // bytes of shared memory required
  float       transfer; // bytes of data transfer per kernel call
  float       transfer2;// bytes of cache line per kernel call
} op_plan;

typedef struct {
  char const *name;     // name of kernel function
  int         count;    // number of times called
  float       time;     // total execution time
  float       transfer; // bytes of data transfer (used)
  float       transfer2;// bytes of data transfer (total)
} op_kernel;

//
//  min / max definitions
//

#ifndef MIN
#define MIN(a,b) ((a<b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a>b) ? (a) : (b))
#endif

//
// alignment macro based on example on page 50 of CUDA Programming Guide version 3.0
// rounds up to nearest multiple of 16 bytes
//

#define ROUND_UP(bytes) (((bytes) + 15) & ~15)

//
// OP function prototypes
//

void op_init(int, char **, int);

void op_decl_const_i(const char* dat, int size, char const *name);

template <class T>
void op_decl_const(int dim, T *dat, char const *name = "")
{
  op_decl_const_i((char*)dat, sizeof(T)*dim, name);
}

void op_fetch_data_i(op_dat<void> *d);

template <class T>
void op_fetch_data(op_dat<T> *d)
{
  op_fetch_data_i((op_dat<void> *)d);
}

void op_diagnostic_output();

void op_timing_output();

void op_exit();

#endif

