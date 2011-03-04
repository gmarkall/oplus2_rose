.. OP2-ROSE documentation master file, created by
   sphinx-quickstart on Fri Mar  4 10:21:10 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

::::::::::::::::::::::::::::::::::::
Welcome to OP2-ROSE's documentation!
::::::::::::::::::::::::::::::::::::

Contents:

.. toctree::
   :maxdepth: 2

#################
API Documentation
#################

The OP2 API consists of two parts we will refer to as the *global API* and the *kernel API* in this document.

The global API is used to write an OP2 program and defines OP2 data types (set, data, map), initialisation and utility function, and the parallel loop call interface.

**************
The Global API
**************

OP2 Data Types
==============

.. cpp:class:: op_set

  A set of entities that can be iterated over in a parallel loop.

  .. cpp:function:: op_set(unsigned int size, vector<int>* partinfo, char const *name = "")

    Set constructor initialising all the data members

  .. cpp:member:: int size

    Total number of elements in the set

  .. cpp:member:: int index

    Global index of this set (in list of sets)

  .. cpp:member:: char const *name
  
    Name of the set

  .. cpp:member:: vector<int>* partinfo
  
    A vector of partition sizes (holds partition info for the set)

.. cpp:class:: op_map 

  A mapping from a *source set* to a *target set*

  .. cpp:function:: op_map(op_set from, op_set to, int dim, int *map, char const *name = "")

    Map constructor initialising all the data members

  .. cpp:member:: op_set from

    Source set mapped from

  .. cpp:member:: op_set to

    Target set mapped to

  .. cpp:member:: int dim

    Target dimension of the mapping

  .. cpp:member:: int index

    Global index of this map (in list of mappings)

  .. cpp:member:: int *map

    Raw data array defining the mapping

  .. cpp:member:: char const *name

    Name of the mapping

.. cpp:class:: template <class T>
  struct op_dat

  .. cpp:function:: op_dat(op_set set, int dim, void* dat, char const *name = "")

    Constructor initialising all the data members

  .. cpp:member:: op_set set
    
    The set on which the data is defined

  .. cpp:member:: int dim
  
    Dimension of the data

  .. cpp:member:: int index
  
    Global index of the datasets (in list of datasets)

  .. cpp:member:: int size
  
    Size of each element in the dataset

  .. cpp:member:: char *dat
  
    Raw data array on the host

  .. cpp:member:: char *dat_d
  
    Raw data array on the device (GPU)

  .. cpp:member:: char *dat_t
  
    Temporary data array on the host

  .. cpp:member:: char const *name
  
    Name of the dataset

  .. cpp:member:: char const *type
  
    Datatype of the dataset

.. cpp:class:: template <class T>
  struct op_dat_gbl : public op_dat<T>

  Global dataset

.. cpp:class:: op_plan

  Parallel execution plan storing colouring information

  *input arguments*

    .. cpp:member:: char const  *name

      Plan identifier

    .. cpp:member:: int          set_index

      Index of the set this plan is associated with

    .. cpp:member:: int          nargs

      Number of arguments to the parallel loop call
      (i.e. number of datasets passed)

    .. cpp:member:: int         *arg_idxs
    .. cpp:member:: int         *idxs
    .. cpp:member:: int         *map_idxs
    .. cpp:member:: int         *dims
    .. cpp:member:: op_access   *accs

      Access descriptors for all arguments to the parallel loop call

  *execution plan*

    .. cpp:member:: int        *nthrcol
    
      number of thread colors for each block

    .. cpp:member:: int        *thrcol
    
      thread colors for each element of the primary set

    .. cpp:member:: int        *offset
    
      offset for primary set for the beginning of each block

    .. cpp:member:: int       **ind_maps
    
      pointers for indirect datasets: 2D array, the outer
      index over the indirect datasets, and the inner one
      giving the local → global renumbering for the
      elements of the indirect set

    .. cpp:member:: int       **ind_offs

      offsets for into ind_maps for each block
      for each indirect dataset

    .. cpp:member:: int       **ind_sizes
    
      indirect indices for each block
      for each indirect dataset

    .. cpp:member:: int       **maps
    
      regular pointers, renumbered as needed

    .. cpp:member:: int        *nelems
    
      number of elements in each block

    .. cpp:member:: int         ncolors
    
      number of block colors

    .. cpp:member:: int        *ncolblk
    
      number of blocks for each color

    .. cpp:member:: int        *blkmap
    
      mapping to blocks of each color

    .. cpp:member:: int         nshared
    
      bytes of shared memory required

    .. cpp:member:: float       transfer
  
      bytes of data transfer per kernel call

    .. cpp:member:: float       transfer2
  
      bytes of cache line per kernel call

.. cpp:class:: op_kernel

  Auxiliary data structure to keep kernel timings and bandwidth statistics

  .. cpp:member:: char const *name
  
    name of kernel function

  .. cpp:member:: int         count
  
    number of times called

  .. cpp:member:: float       time
  
    total execution time

  .. cpp:member:: float       transfer
  
    bytes of data transfer (used)

  .. cpp:member:: float       transfer2
  
    bytes of data transfer (total)


Initialisation and Termination
==============================

.. cpp:function:: void op_init( int argc, char **argv, int diags_level)

  This function must be called before all other OP routines.

  :param argc: number of command line arguments
  :param argv: the usual command line arguments
  :param diags_level: an integer which defines the level of debugging diagnostics and reporting to be performed

    |
    | 0 -- none
    | 1 -- error-checking
    | 2 -- info on plan construction
    | 3 -- report execution of parallel loops
    | 4 -- report use of old plans
    | 7 -- report positive checks in ``op_plan_check``

.. cpp:function:: void op_exit()

  This function must be called last to cleanly terminate the OP computation.

.. cpp:function:: template <class T>
  void op_decl_const(int dim, T *dat, char const *name = "")

  This function declares constant data with global scope to be used in user's kernel functions.

  Note: in sequential version, it is the user's responsibility to define the appropriate global variable.

Diagnostic and utility
======================

.. cpp:function:: void op_diagnostic_output()

  This routine prints out various useful bits of diagnostic info about sets, mappings and datasets

.. cpp:function:: void op_timing_output()

  This function prints runtime and bandwidth statistics for kernels

.. cpp:function:: template <class T>
  void op_fetch_data(op_dat<T> *d)

  Fetch a dataset from the device

Parallel Loop Execution
=======================

As an example, the parallel loop syntax when the user's kernel function has 3 arguments, with the third being a local constant or global reduction array, is:

.. cpp:function:: template < class T0, class T1, class T2 >
  float op_par_loop(void (*kernel)( T0*, T1*, T2* ),
  op_set set,                                                         
  op_dat<T0> *arg0 ,int idx0 ,op_map *map0 ,op_access acc0 ,          
  op_dat<T1> *arg1 ,int idx1 ,op_map *map1 ,op_access acc1 ,
  op_dat<T2> *arg2 ,int idx2 ,op_map *map2 ,op_access acc2)

  :param kernel:     user's kernel function with 3 arguments of arbitrary type (this is only used for the single-threaded CPU build)
  :param name:       name of kernel function, used for output diagnostics
  :param set:        OP set ID, giving set over which the parallel computation is performed
  :param arg:        OP dataset ID, or pointer to constant or global reduction array
  :param idx:        index of mapping to be used (-1 $\equiv$ no mapping indirection)
  :param map:        OP mapping ID ({\tt OP\_ID} for identity mapping, i.e.~no mapping indirection,
                  {\tt OP\_GBL} for constant or global reduction array)
  :param acc:        access type:

    |
    | ``OP_READ``: read-only
    | ``OP_WRITE``: write-only, but without potential data conflict
    | ``OP_RW``:  read and write, but without potential data conflict
    | ``OP_INC``: increment, or global reduction to compute a sum
    | ``OP_MAX``: global reduction to compute a maximum
    | ``OP_MIN``: global reduction to compute a minimum

  In this example, ``kernel`` is a function with 3 arguments of arbitrary type which performs a calculation for a single set element.  This will get converted by a preprocessor into a routine called by the CUDA kernel function.  The preprocessor will also take the specification of the arguments and turn this into the CUDA kernel function which loads in indirect data (i.e.~data addressed indirectly through a mapping) from the device main memory into the shared storage, then calls the converted ``kernel`` function for each element for each line in the above specification.  Indirect data is incremented in shared memory (with thread coloring to avoid possible data conflicts) before being updated at the end of the CUDA kernel call.
