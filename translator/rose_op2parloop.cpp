/*
 * 
 * Copyright (c) 2010, Graham Markall
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimer in the documentation
 *       and/or other materials provided with the distribution.
 *     * Neither the name of Imperial College London nor the names of its
 *       contributors may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */

#include <iostream>
#include <sstream>

#include "rose_op2parloop.h"


const int op_par_loop_args::num_params = 2;



/////////// Utility string functions for creating argument names ///////////////

inline string buildStr(int i)
{
  stringstream s;
  s << i;
  return s.str();
}

inline string arg(int i)
{
  return "arg"+buildStr(i);
}

inline string argLocal(int i)
{
  return arg(i)+"_l";
}

///////// Utility functions that may be moved to another file later

SgFunctionCallExp* isOpParLoop(SgNode *n)
{
  SgFunctionCallExp *fn = isSgFunctionCallExp(n);
  if (fn)
  {
    string fn_name = fn->getAssociatedFunctionDeclaration()->get_name().getString();
    if (  fn_name.compare("op_par_loop_2")==0
       || fn_name.compare("op_par_loop_3")==0
       || fn_name.compare("op_par_loop_4")==0 
       || fn_name.compare("op_par_loop_5")==0
       || fn_name.compare("op_par_loop_6")==0
       || fn_name.compare("op_par_loop_7")==0
       || fn_name.compare("op_par_loop_8")==0
       || fn_name.compare("op_par_loop_9")==0
      )
    {
      cerr << "Located Function " << fn_name << endl;
      return fn;
    }
  }
  
  return NULL;
}


/////////// op_par_loop_args : Implementation //////////////////////////////////

void op_par_loop_args::init(SgFunctionCallExp* fn)
{
  SgExpressionPtrList& fnCallArgs = fn->get_args()->get_expressions();
      
  // We parse the arguments to the op_par_loop_3 call into our internal 
  // representation that is a little more convenient for later on.
  SgExpressionPtrList::iterator i=fnCallArgs.begin();
  kernel = isSgFunctionRefExp(*i);
  ++i;
  set = isSgVarRefExp(*i);
  ++i;
  
  // Calculate number of args = total - 3 i.e. kernel, label and set and create arg objects
  int numArgs = (fnCallArgs.size() - num_params) / op_argument::num_params;
  for (int j=0; j<numArgs; ++j)
  {
    op_argument* parg = new op_argument(i);
    parg->own_index = args.size();
    updatePlanContainer(parg);
    
    args.push_back(parg);
    if (parg->usesIndirection())
    {
      ind_args.push_back(parg);
    }
  }
}

void op_par_loop_args::updatePlanContainer(op_argument* argument)
{
  if(argument->usesIndirection())
  {
    string cur_name = argument->arg->get_symbol()->get_name().getString();
    if(prev_name.compare(cur_name) != 0)
    {
        planContainer.push_back(argument);
        prev_name = cur_name;
    }
    argument->plan_index = planContainer.size()-1;
  }
}


//////////// OPParLoop : Implementation ////////////////////////////////////////

/* 
 * ParLoop needs to know which ROSE project it is working on
 */
void OPParLoop::setProject(SgProject *p)
{
  project = p;
}

/*
 * The visit function is reimplemented from AstSimpleProcessing, and is called
 * for every node in the the AST. We are only really interested in looking at
 * function calls to op_par_loop_3.
 */
void OPParLoop::visit(SgNode *n) 
{
  SgGlobal *globalScope;
  SgFunctionCallExp *fn;
  
  // We need to put the global scope on the scope stack so that we can look
  // up the oplus datatypes later on (in generateSpecial).
  if ((globalScope = isSgGlobal(n)))
  {
    pushScopeStack(globalScope);
  }

  if ((fn = isOpParLoop(n)))
  {
    op_par_loop_args *parLoopArgs = new op_par_loop_args();
    parLoopArgs->init(fn);
    
    // Generate kernels
    if(parLoopArgs->numIndArgs() == 0)
    {
      generateSpecial(fn, parLoopArgs);
    }
    else
    {
      generateStandard(fn, parLoopArgs);
    }
  }
}

/* 
 * Outputs each generated kernel file to disk.
 */
void OPParLoop::unparse()
{
  for(vector<SgProject*>::iterator i=kernels.begin(); i!=kernels.end(); ++i)
  {  
    cerr << "Running AST tests." << endl;
    AstTests::runAllTests(*i);
    cerr << "AST tests passed." <<endl;
    (*i)->unparse();
    cerr << "Unparsed." << endl;
  }
}

/*
 *  Generate global kernel header
 */
void OPParLoop::generateGlobalKernelsHeader()
{
  // We build a new file for the CUDA kernel and its stub function
  string file_name = "kernels.h";
  cerr << "Generating CUDA Kernels File" << endl;
  
  SgSourceFile *file = isSgSourceFile(buildFile("blank.cpp", file_name, NULL));
  ROSE_ASSERT(file!=NULL);
  SgGlobal *globalScope = file->get_globalScope();
  
  addTextForUnparser(globalScope, "#ifndef OP_KERNELS\n", AstUnparseAttribute::e_before);
  addTextForUnparser(globalScope, "#define OP_KERNELS\n", AstUnparseAttribute::e_before);
  addTextForUnparser(globalScope, "#include \"op_datatypes.h\"\n", AstUnparseAttribute::e_before);
  
  map<string, SgFunctionDeclaration*>::iterator p;
  for(p=cudaFunctionDeclarations.begin(); p != cudaFunctionDeclarations.end() ; ++p)
  {
    SgFunctionDeclaration* copy = (*p).second;
    SgFunctionDeclaration* d = buildNondefiningFunctionDeclaration(copy, globalScope);
    appendStatement(d,globalScope);
  }
  
  addTextForUnparser(globalScope, "#endif\n", AstUnparseAttribute::e_after);
  

  // Add to list of files that need to be unparsed.
  kernels.push_back(file->get_project());
}

/*
 * Returns the name of the function pointed to by a function pointer.
 */
string OPParLoop::getName(SgFunctionRefExp *fn)
{
  return fn->get_symbol_i()->get_name().getString();
}

/* 
 * Print out debug info at traversal end
 */
void OPParLoop::atTraversalEnd() 
{
  cerr << "Traversal Ended." << endl;
}


void OPParLoop::createKernelFile(string kernel_name)
{  
  // We build a new file for the CUDA kernel and its stub function
  string file_name = kernel_name + "_kernel.cu";
  cerr << "Generating (Special) CUDA Parallel Loop File for " << kernel_name << endl;
  
  SgSourceFile *file = isSgSourceFile(buildFile("blank.cpp", file_name, NULL));
  ROSE_ASSERT(file!=NULL);
  fileGlobalScope = file->get_globalScope();
  
  addTextForUnparser(fileGlobalScope,"#include \"user_defined_types.h\"\n",AstUnparseAttribute::e_before);
  addTextForUnparser(fileGlobalScope,"#include \"op_datatypes.h\"\n",AstUnparseAttribute::e_before);
  addTextForUnparser(fileGlobalScope,"#include \"kernels.h\"\n\n",AstUnparseAttribute::e_before);
  
  // C kernel prefixed with the __device__ keyword. However, we could copy the AST
  // subtree representing the kernel into the new file, which would remove the
  // requirement for writing each kernel in a separate .h file.
  addTextForUnparser(fileGlobalScope, "\n\n__device__\n#include <"+kernel_name+".h>\n",AstUnparseAttribute::e_before);

  // Add to list of files that need to be unparsed.
  kernels.push_back(file->get_project());
}  

void OPParLoop::initialiseDataTypes()
{  
  // In order to build the prototype for the plan function, we need to get hold of the types 
  // that we intend to pass it. Since these are declared in op_datatypes.h, we need to 
  // loop them up before we can use them.
  op_set = lookupNamedTypeInParentScopes("op_set");
  op_dat = SgClassType::createType(buildStructDeclaration(SgName("op_dat<void>"), fileGlobalScope));
  op_ptr = lookupNamedTypeInParentScopes("op_ptr");
  op_access = lookupNamedTypeInParentScopes("op_access");
  op_plan = lookupNamedTypeInParentScopes("op_plan");
}

void OPParLoop::createKernel(string kernel_name, SgFunctionParameterList *paramList)
{
  // We can build the __global__ function using the parameter list and add it to our new file. We get a reference to
  // the body of this function so that we can add code to it later on.
  SgFunctionDeclaration *func = buildDefiningFunctionDeclaration("op_cuda_"+kernel_name, buildVoidType(), paramList, fileGlobalScope);
  addTextForUnparser(func,"\n\n__global__",AstUnparseAttribute::e_before);
  appendStatement(func, fileGlobalScope);
  kernelBody = func->get_definition()->get_body();
}

void OPParLoop::createStub(string kernel_name, SgFunctionParameterList *paramList)
{
  // We build the function with the parameter list and insert it into the global
  // scope of our file as before.
  string stubName = "op_par_loop_" + kernel_name;
  SgFunctionDeclaration *stubFunc = buildDefiningFunctionDeclaration(stubName, buildFloatType(), paramList, fileGlobalScope);
  cudaFunctionDeclarations.insert(pair<string, SgFunctionDeclaration*>(kernel_name, stubFunc));
  appendStatement(stubFunc, fileGlobalScope);
  stubBody = stubFunc->get_definition()->get_body();
}

void OPParLoop::createReductionKernel(string kernel_name, SgFunctionParameterList *paramList)
{
  // This will be better refactored with createKernel, but that's less pressing now.
  SgFunctionDeclaration *func = buildDefiningFunctionDeclaration("op_cuda_"+kernel_name+"_reduction", buildVoidType(), paramList, fileGlobalScope);
  addTextForUnparser(func,"\n\n__global__",AstUnparseAttribute::e_before);
  appendStatement(func, fileGlobalScope);
  reductionBody = func->get_definition()->get_body();
}

void OPParLoop::createSharedVariable(string name, SgType *type, SgScopeStatement *scope)
{
  SgVariableDeclaration *varDec = buildVariableDeclaration(name, type, NULL, scope);
  addTextForUnparser(varDec,"\n\n  __shared__ ", AstUnparseAttribute::e_after);
  appendStatement(varDec, scope);
}

void OPParLoop::createBeginTimerBlock(SgScopeStatement *scope)
{
  // Add the timer block
  //-----------------------------------------
  /*
  float elapsed_time_ms=0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop  );
  cudaEventRecord( start, 0 );
  */
  //-----------------------------------------
  SgVariableDeclaration *varDec;
  SgExprStatement *fnCall;
  varDec = buildVariableDeclaration(SgName("elapsed_time_ms"), buildFloatType(), buildAssignInitializer(buildFloatVal(0.0f)), stubBody);
  addTextForUnparser(varDec,"\ncudaEvent_t start, stop;", AstUnparseAttribute::e_after);
  appendStatement(varDec,scope);
  fnCall = buildFunctionCallStmt("cudaEventCreate", buildVoidType(), buildExprListExp(buildAddressOfOp(buildOpaqueVarRefExp(SgName("start")))), stubBody);
  appendStatement(fnCall,scope);
  fnCall = buildFunctionCallStmt("cudaEventCreate", buildVoidType(), buildExprListExp(buildAddressOfOp(buildOpaqueVarRefExp(SgName("stop")))), stubBody);
  appendStatement(fnCall,scope);
  fnCall = buildFunctionCallStmt("cudaEventRecord", buildVoidType(), buildExprListExp(buildOpaqueVarRefExp(SgName("start")), buildIntVal(0)), stubBody);
  appendStatement(fnCall,scope);
}

void OPParLoop::createEndTimerBlock(SgScopeStatement *scope, bool accumulateTime)
{
  // Add the timer block
  //------------------------------------------------------
  /*
  cudaEventRecord( stop, 0 );
  cudaThreadSynchronize();
  cudaEventElapsedTime( &elapsed_time_ms, start, stop );
  cudaEventDestroy( start );
  cudaEventDestroy( stop );
  total_time += elapsed_time_ms;
  */  
  //------------------------------------------------------
  SgExprStatement *fnCall;
  fnCall = buildFunctionCallStmt("cudaEventRecord", buildVoidType(), buildExprListExp(buildOpaqueVarRefExp(SgName("stop")), buildIntVal(0)), stubBody);
  appendStatement(fnCall,scope);
  fnCall = buildFunctionCallStmt("cudaThreadSynchronize", buildVoidType(), NULL, stubBody);
  appendStatement(fnCall, scope);
  fnCall = buildFunctionCallStmt("cudaEventElapsedTime", buildVoidType(), buildExprListExp( buildOpaqueVarRefExp(SgName("&elapsed_time_ms")), buildOpaqueVarRefExp(SgName("start")),
  buildOpaqueVarRefExp(SgName("stop")) ), stubBody);
  appendStatement(fnCall, scope);
  fnCall = buildFunctionCallStmt("cudaEventDestroy", buildVoidType(), buildExprListExp( buildOpaqueVarRefExp(SgName("start")) ), stubBody);
  appendStatement(fnCall, scope);
  fnCall = buildFunctionCallStmt("cudaEventDestroy", buildVoidType(), buildExprListExp( buildOpaqueVarRefExp(SgName("stop")) ), stubBody);
  appendStatement(fnCall, scope);

  if (accumulateTime)
  {
    SgPlusAssignOp *e = buildPlusAssignOp( buildOpaqueVarRefExp(SgName("total_time")), buildOpaqueVarRefExp(SgName("elapsed_time_ms")) );
    appendStatement(buildExprStatement(e),scope);  
  }
}

SgFunctionParameterList* OPParLoop::createSpecialParameters(op_par_loop_args *pl)
{
  // We need to build a list of parameters for our __global__ function,
  // based on the arguments given to op_par_loop_3 earlier:
  SgFunctionParameterList *paramList = buildFunctionParameterList();
  reduction_required = false;
  for(int i=0; i<pl->numArgs(); i++)
  {
    if(pl->args[i]->consideredAsReduction())
      reduction_required = true;
    SgInitializedName *name;
    SgType *argType = buildPointerType(pl->args[i]->type);
    name = buildInitializedName(arg(i), argType);
    paramList->append_arg(name);
  }
  SgInitializedName *set_size = buildInitializedName("set_size", buildIntType());
  appendArg(paramList, set_size);
  for(int i=0; i<pl->numArgs(); i++)
  {
    if(pl->args[i]->consideredAsReduction())
    {
      SgInitializedName *block_reduct = buildInitializedName("block_reduct" + buildStr(i), buildPointerType(buildVoidType()));
      appendArg(paramList, block_reduct);
    }
  }

  return paramList;
}

SgFunctionParameterList* OPParLoop::createStandardParameters(op_par_loop_args *pl)
{
  SgFunctionParameterList *paramList = buildFunctionParameterList();
  SgType *argType = NULL;
  SgInitializedName *nm = NULL;
  
  // First Assemble all expressions using plan container <for arguments with indirection>
  for(unsigned int i=0; i<pl->planContainer.size(); i++)
  {
    // Add "ind_arg0"
    argType = buildPointerType(pl->planContainer[i]->type);
    nm = buildInitializedName(SgName("ind_arg" + buildStr(i)), argType);
    appendArg(paramList, nm);

    // Add "ind_arg0_ptrs"
    argType = buildPointerType(buildIntType());
    nm = buildInitializedName(SgName("ind_arg" + buildStr(i) + "_ptrs"), argType);
    appendArg(paramList, nm);

    // Add "ind_arg0_sizes"
    nm = buildInitializedName(SgName("ind_arg" + buildStr(i) + "_sizes"), argType);
    appendArg(paramList, nm);
  
    // Add "ind_arg0_offset"
    nm = buildInitializedName(SgName("ind_arg" + buildStr(i) + "_offset"), argType);
    appendArg(paramList, nm);
  }
  // Then add all the pointers
  reduction_required = false;
  for(unsigned int i=0; i<pl->args.size(); i++)
  {
    if(pl->args[i]->consideredAsReduction())
      reduction_required = true;
    if(pl->args[i]->usesIndirection())
    {
      // Add "arg1_ptr"
      argType = buildPointerType(buildIntType());
      nm = buildInitializedName(arg(i) + SgName("_ptrs"), argType);
      appendArg(paramList, nm);
    }
    else if(pl->args[i]->isGlobal())
    {
      argType = buildPointerType(pl->args[i]->type);
      nm = buildInitializedName(arg(i), argType);
      appendArg(paramList, nm);
    }
    else
    {
      argType = buildPointerType(pl->args[i]->type);
      nm = buildInitializedName(arg(i) + SgName("_d"), argType);
      appendArg(paramList, nm);
    }
  }
  // Other stuff
  argType = buildIntType();
  nm = buildInitializedName(SgName("block_offset"), argType);
  appendArg(paramList, nm);
  argType = buildPointerType(argType);
  nm = buildInitializedName(SgName("blkmap"), argType);
  appendArg(paramList, nm);
  nm = buildInitializedName(SgName("offset"), argType);
  appendArg(paramList, nm);
  nm = buildInitializedName(SgName("nelems"), argType);
  appendArg(paramList, nm);
  nm = buildInitializedName(SgName("ncolors"), argType);
  appendArg(paramList, nm);
  nm = buildInitializedName(SgName("colors"), argType);
  appendArg(paramList, nm);
  if(reduction_required)
    appendArg(paramList, buildInitializedName(SgName("block_reduct"), buildVoidType()));

  return paramList;
}

void OPParLoop::createSharedVariableDeclarations(op_par_loop_args *pl)
{
  // Add shared memory declaration
  SgVariableDeclaration *varDec = buildVariableDeclaration(SgName("shared"), buildArrayType(buildCharType(), NULL), NULL, kernelBody);
  addTextForUnparser(varDec,"\n\n  extern __shared__ ", AstUnparseAttribute::e_after);
  appendStatement(varDec,kernelBody);

  // Add shared variables for the planContainer variables - for each category <ptr>
  for(unsigned int i=0; i<pl->planContainer.size(); i++)
  {
    createSharedVariable("ind_"+arg(i)+"_ptr", buildPointerType(buildIntType()), kernelBody);
  }

  // Add shared variables for the planContainer variables - for each category <size>
  for(unsigned int i=0; i<pl->planContainer.size(); i++)
  {
    createSharedVariable("ind_"+arg(i)+"_size", buildIntType(), kernelBody);
  }

  // Add shared variables for the planContainer variables - for each category <s for shared>
  for(unsigned int i=0; i<pl->planContainer.size(); i++)
  {
    SgType *t = buildPointerType(pl->planContainer[i]->type);
    createSharedVariable("ind_"+arg(i)+"_s", t, kernelBody);
  }

  // Then add respective shared variables for each argument
  for(unsigned int i=0; i<pl->args.size(); i++)
  {
    if(pl->args[i]->usesIndirection())
    {
      createSharedVariable(arg(i)+"_ptr", buildPointerType(buildIntType()), kernelBody);
    }
    else if(!pl->args[i]->consideredAsConst())
    {
      SgType *t = buildPointerType(pl->args[i]->type);
      createSharedVariable(arg(i), t, kernelBody);
    }
  }

  createSharedVariable("nelem2", buildIntType(), kernelBody);
  createSharedVariable("ncolor", buildIntType(), kernelBody);
  createSharedVariable("color", buildPointerType(buildIntType()), kernelBody);
  createSharedVariable("blockId", buildIntType(), kernelBody);
  createSharedVariable("nelem", buildIntType(), kernelBody);
}

/*
 *  Generate Seperate File For the Special Kernel
 *  ---------------------------------------------
 */ 
void OPParLoop::generateSpecial(SgFunctionCallExp *fn, op_par_loop_args *pl)
{
  string kernel_name = getName(pl->kernel);
  createKernelFile(kernel_name);

  initialiseDataTypes();
 
  SgFunctionParameterList *paramList = createSpecialParameters(pl);
 
  createKernel(kernel_name, paramList);

  // We Add the declarations of local variables first.
  for(int i=0; i<pl->numArgs(); i++)
  {
    if(pl->args[i]->isGlobal())
    {
      SgVariableDeclaration *varDec;
      varDec = buildVariableDeclaration(argLocal(i), buildArrayType(pl->args[i]->type, buildIntVal(pl->args[i]->dim)), NULL, kernelBody);
      appendStatement(varDec,kernelBody);
    }
  }

  preKernelGlobalDataHandling(fn, pl);

  // 3 MAIN EXECUTION LOOP <BEGIN>
  // =======================================
  SgScopeStatement *loopBody = buildBasicBlock();
  SgExpression *rhs = buildAddOp(buildOpaqueVarRefExp("threadIdx.x"), buildMultiplyOp(buildOpaqueVarRefExp("blockIdx.x"), buildOpaqueVarRefExp("blockDim.x")));
  SgVariableDeclaration *loopVarDec = buildVariableDeclaration(SgName("n"), buildIntType(), buildAssignInitializer(rhs), loopBody);
  SgName loopVar = loopVarDec->get_definition()->get_vardefn()->get_name();

  SgExpression *lhs = buildVarRefExp(loopVar, loopBody);
  rhs = buildOpaqueVarRefExp(SgName("set_size"));
  SgExprStatement *test = buildExprStatement(buildLessThanOp(lhs, rhs));

  rhs = buildMultiplyOp(buildOpaqueVarRefExp("blockDim.x"), buildOpaqueVarRefExp("gridDim.x"));
  SgPlusAssignOp *increment = buildPlusAssignOp(lhs, rhs);
  SgForStatement *forLoop = buildForStatement(loopVarDec, test, increment, loopBody);

  // 3.1 FIRE KERNEL
  // ===============================================
  // Next we build a call to the __device__ function. We build the parameters
  // for this call first, then the call, and add it into the outer loop body.
  SgExprListExp *kPars = buildExprListExp();
  for(int i=0; i<pl->numArgs(); i++)
  {
    if(pl->args[i]->isNotGlobal())
    {
      SgExpression *e = buildAddOp(buildOpaqueVarRefExp(arg(i)), buildMultiplyOp(buildOpaqueVarRefExp(loopVar), buildIntVal(pl->args[i]->dim)));
      kPars->append_expression(e);
    }
    else
    {
      SgExpression *e = buildOpaqueVarRefExp(argLocal(i), kernelBody);
      kPars->append_expression(e);
    }
  }
  SgExprStatement *uf = buildFunctionCallStmt(SgName(kernel_name), buildVoidType(), kPars);
  appendStatement(uf,loopBody);

  // 3 MAIN EXECUTION LOOP <END>
  // =======================================
  // Now we have completed the body of the outer for loop, we can build an initialiser, 
  // an increment and a test statement. The we insert this loop into the __gloabl__ function.
  // Because threadIdx.x etc are not really variables, we invent "opaque" variables with these
  // names.
  appendStatement(forLoop,kernelBody);
  
  postKernelGlobalDataHandling(fn, pl);

  if (reduction_required)
  {
    generateReductionKernel(kernel_name, pl);
  }

  generateSpecialStub(fn, kernel_name, pl);
}

SgFunctionParameterList* OPParLoop::createReductionParameters(op_par_loop_args *pl)
{
  SgFunctionParameterList *paramList = buildFunctionParameterList();
  SgInitializedName *grid_size = buildInitializedName("gridsize", buildIntType());
  appendArg(paramList, grid_size);
  for(int i=0; i<pl->numArgs(); i++)
  {
    if(pl->args[i]->consideredAsReduction())
    {
      SgType *argType = buildPointerType(pl->args[i]->type);
      SgInitializedName *name = buildInitializedName(arg(i), argType);
      paramList->append_arg(name);

      SgInitializedName *block_reduct = buildInitializedName("block_reduct" + buildStr(i), buildPointerType(buildVoidType()));
      appendArg(paramList, block_reduct);
    }
  }

  return paramList;
}


void OPParLoop::generateReductionKernel(string kernel_name, op_par_loop_args *pl)
{
  SgFunctionParameterList *paramList = createReductionParameters(pl);
 
  createReductionKernel(kernel_name, paramList);

  for(int i=0; i<pl->numArgs(); i++)
  {
    if(pl->args[i]->consideredAsReduction())
    {
      // Create loop
      SgStatement *viInit = buildVariableDeclaration(SgName("d"), buildIntType(), buildAssignInitializer(buildIntVal(0)));
      SgName indVar = isSgVariableDeclaration(viInit)->get_definition()->get_vardefn()->get_name();
	    
      // Call function
      SgExprListExp *kPars2 = buildExprListExp();
      SgExpression *e = buildOpaqueVarRefExp(arg(i));
      e = buildAddOp(e, buildVarRefExp(indVar));
      kPars2->append_expression(e);
      e = buildOpaqueVarRefExp(SgName("block_reduct" + buildStr(i)));
      kPars2->append_expression(e);
      e = buildOpaqueVarRefExp(SgName("gridsize"));
      kPars2->append_expression(e);

      SgExprStatement *uf1 = NULL;
      switch(pl->args[i]->access)
      {
	case OP_INC:
	  uf1 = buildFunctionCallStmt(SgName("op_reduction2_2<OP_INC>"), buildVoidType(), kPars2);
	  break;
	case OP_MAX:
	  uf1 = buildFunctionCallStmt(SgName("op_reduction2_2<OP_MAX>"), buildVoidType(), kPars2);
	  break;
	case OP_MIN:
	  uf1 = buildFunctionCallStmt(SgName("op_reduction2_2<OP_MIN>"), buildVoidType(), kPars2);
	  break;
	default:
	  break;
      }

      // build test and increment, and add the loop into the body of the inner loop.
      SgScopeStatement *viLoopBody = buildBasicBlock(uf1);
      SgExprStatement *viTest = buildExprStatement(buildLessThanOp(buildOpaqueVarRefExp(indVar), buildIntVal(pl->args[i]->dim)));
      SgPlusPlusOp *viIncrement = buildPlusPlusOp(buildOpaqueVarRefExp(indVar));
      SgStatement *viForLoop = buildForStatement(viInit, viTest, viIncrement, viLoopBody);
      appendStatement(viForLoop,reductionBody);
    }
  }
}

SgFunctionParameterList* OPParLoop::buildSpecialStubParameters(op_par_loop_args *pl)
{
  SgFunctionParameterList *paramList = buildFunctionParameterList();
  SgInitializedName *name = buildInitializedName(SgName("name"), buildPointerType(buildConstType(buildCharType())));
  appendArg(paramList, name);
  name = buildInitializedName(SgName("set"), op_set);
  appendArg(paramList, name);
  // Add other arguments
  for(int i=0; i<pl->numArgs(); i++)
  {
    name = buildInitializedName(SgName("arg"+buildStr(i)), buildPointerType(op_dat));
    appendArg(paramList, name);
    name = buildInitializedName(SgName("idx"+buildStr(i)), buildIntType());
    appendArg(paramList, name);
    name = buildInitializedName(SgName("ptr"+buildStr(i)), buildPointerType(op_ptr));
    appendArg(paramList, name);
    name = buildInitializedName(SgName("acc"+buildStr(i)), op_access);
    appendArg(paramList, name);
  }
  
  return paramList;
}

void OPParLoop::generateSpecialStub(SgFunctionCallExp *fn, string kernel_name, op_par_loop_args *pl)
{
  SgFunctionParameterList *paramList = buildSpecialStubParameters(pl);
  createStub(kernel_name, paramList);

  createSpecialStubVariables();

  preHandleConstAndGlobalData(fn, pl);

  createBeginTimerBlock(stubBody);

  createSpecialKernelCall(kernel_name, pl);

  if (reduction_required)
  {
    createReductionKernelCall(kernel_name, pl);
  }

  createEndTimerBlock(stubBody, false);

  postHandleConstAndGlobalData(fn, pl);

  SgReturnStmt* rtstmt = buildReturnStmt(buildOpaqueVarRefExp(SgName("elapsed_time_ms")));
  appendStatement(rtstmt, stubBody);
  
}

void OPParLoop::createSpecialStubVariables()
{
  // Declare gridsize and bsize
  SgExpression *e = buildOpaqueVarRefExp(SgName("BSIZE"));
  SgVariableDeclaration *varDec = buildVariableDeclaration(SgName("bsize"), buildIntType(), buildAssignInitializer(e));
  appendStatement(varDec, stubBody);
  
  e = buildOpaqueVarRefExp(SgName("set.size"));
  e = buildSubtractOp(e, buildIntVal(1));
  e = buildDivideOp(e, buildOpaqueVarRefExp(SgName("bsize")));
  e = buildAddOp(e, buildIntVal(1));
  varDec = buildVariableDeclaration(SgName("gridsize"), buildIntType(), buildAssignInitializer(e));
  appendStatement(varDec, stubBody);
}


void OPParLoop::createSpecialKernelCall(string kernel_name, op_par_loop_args *pl)
{
  // To add a call to the CUDA function, we need to build a list of parameters that
  // we pass to it. The easiest way to do this is to name the members of the 
  // struct to which they belong, but this is not the most elegant approach.
  SgExprListExp *kPars = buildExprListExp();
  for(int i=0; i<pl->numArgs(); i++)
  {
    SgExpression *e = buildOpaqueVarRefExp(SgName("arg"+buildStr(i)+"->dat_d"));
    SgCastExp* e_cast = buildCastExp(e, buildPointerType(pl->args[i]->type));
    kPars->append_expression(e_cast);
  }
  SgExpression *e = buildOpaqueVarRefExp(SgName("set.size"));
  kPars->append_expression(e);
  for(int i=0; i<pl->numArgs(); i++)
  {
    if(pl->args[i]->consideredAsReduction())
    {
      kPars->append_expression(buildOpaqueVarRefExp(SgName("block_reduct" + buildStr(i))));
    }
  }

  // We have to add the kernel configuration as part of the function name
  // as CUDA is not directly supported by ROSE - however, I understand
  // that CUDA and OpenCL support is coming soon!
  SgExprStatement *kCall = buildFunctionCallStmt("op_cuda_"+kernel_name+"<<<gridsize,bsize,reduct_shared>>>", buildVoidType(), kPars, stubBody);
  appendStatement(kCall,stubBody);
}

void OPParLoop::createReductionKernelCall(string kernel_name, op_par_loop_args *pl)
{
  SgExprListExp *kPars = buildExprListExp();
  kPars->append_expression(buildOpaqueVarRefExp(SgName("gridsize")));
  for(int i=0; i<pl->numArgs(); i++)
  {
    if(pl->args[i]->consideredAsReduction())
    {
      SgExpression *e = buildOpaqueVarRefExp(SgName("arg"+buildStr(i)+"->dat_d"));
      SgCastExp* e_cast = buildCastExp(e, buildPointerType(pl->args[i]->type));
      kPars->append_expression(e_cast);

      kPars->append_expression(buildOpaqueVarRefExp(SgName("block_reduct" + buildStr(i))));
    }
  }
  SgExprStatement *kCall = buildFunctionCallStmt("op_cuda_"+kernel_name+"_reduction<<<1,1,reduct_shared>>>", buildVoidType(), kPars, stubBody);
  appendStatement(kCall,stubBody);
}

/*
 *  Generate Seperate File For the Standard Kernel
 *  ----------------------------------------------
 */ 
void OPParLoop::generateStandard(SgFunctionCallExp *fn, op_par_loop_args *pl)
{
  string kernel_name = getName(pl->kernel);
  createKernelFile(kernel_name);
  
  initialiseDataTypes();
 
  SgFunctionParameterList *paramList = createStandardParameters(pl);

  createKernel(kernel_name, paramList);

  // 2. ADD DECLARATION OF LOCAL VARIABLES
  // ===============================================

  // We Add the declarations of local variables first, required only for INC
  for(int i=0; i<pl->numArgs(); i++)
  {
    if((pl->args[i]->isNotGlobal() && pl->args[i]->access == OP_INC) || (pl->args[i]->isNotGlobal() && !pl->args[i]->usesIndirection()))
    {
      SgType *argType = pl->args[i]->type;
      SgVariableDeclaration *varDec = buildVariableDeclaration(argLocal(i), buildArrayType(argType, buildIntVal(pl->args[i]->dim)), NULL, kernelBody);
      appendStatement(varDec,kernelBody);
    }
  }

  createSharedVariableDeclarations(pl);
  
  // 4.1 GET SIZES AND SHIFT POINTERS AND DIRECT MAPPED DATA
  // ========================================================

  // We put this part within an IF condition, so that threadIdx.x == 0 performs this
  SgScopeStatement *ifBody = buildBasicBlock();
  SgExprStatement *conditionStmt = buildExprStatement( buildEqualityOp( buildOpaqueVarRefExp(SgName("threadIdx.x")), buildIntVal(0) ) );
  SgIfStmt* threadCondition = buildIfStmt(conditionStmt, ifBody, NULL);

  // Add blockId variable
  SgExpression* expression = buildOpaqueVarRefExp(SgName("blkmap[blockIdx.x + block_offset]"));
  expression = buildAssignOp(buildOpaqueVarRefExp(SgName("blockId")), expression);
  appendStatement(buildExprStatement(expression), ifBody);

  // Add blockId variable
  expression = buildOpaqueVarRefExp(SgName("nelems[blockId]"));
  expression = buildAssignOp(buildOpaqueVarRefExp(SgName("nelem")), expression);
  appendStatement(buildExprStatement(expression), ifBody);

  // Add ncolor variable
  expression = buildOpaqueVarRefExp(SgName("ncolors[blockId]"));
  expression = buildAssignOp(buildOpaqueVarRefExp(SgName("ncolor")), expression);
  appendStatement(buildExprStatement(expression), ifBody);

  // Cache offset[blockId]
  expression = buildOpaqueVarRefExp(SgName("offset[blockId]"));
  SgVariableDeclaration* varDec = buildVariableDeclaration(SgName("cur_offset"), buildIntType(), buildAssignInitializer(expression), ifBody);
  appendStatement(varDec, ifBody);

  // Add color variable
  expression = buildOpaqueVarRefExp(SgName("cur_offset"));
  expression = buildAddOp(buildOpaqueVarRefExp(SgName("colors")), expression);
  expression = buildAssignOp(buildOpaqueVarRefExp(SgName("color")), expression);
  appendStatement(buildExprStatement(expression), ifBody);

  // Example : int nelem2 = blockDim.x*(1+(X)/blockDim.x);
  expression = buildOpaqueVarRefExp(SgName("nelem"));
  expression = buildSubtractOp(expression, buildIntVal(1));
  expression = buildDivideOp(expression, buildOpaqueVarRefExp(SgName("blockDim.x")));
  expression = buildAddOp(buildIntVal(1), expression);
  expression = buildMultiplyOp(buildOpaqueVarRefExp(SgName("blockDim.x")), expression);
  expression = buildAssignOp(buildOpaqueVarRefExp(SgName("nelem2")), expression);
  appendStatement(buildExprStatement(expression), ifBody);

  // Calculate the category sizes
  for(unsigned int i=0; i<pl->planContainer.size(); i++)
  {
    expression = buildOpaqueVarRefExp(SgName("ind_" + arg(i) + "_sizes[blockId]"));
    expression = buildAssignOp(buildOpaqueVarRefExp(SgName("ind_" + arg(i) + "_size")), expression);
    appendStatement(buildExprStatement(expression), ifBody);
  }

  // Calculate the category pointers
  for(unsigned int i=0; i<pl->planContainer.size(); i++)
  {
    expression = buildOpaqueVarRefExp(SgName("ind_" + arg(i) + "_offset[blockId]"));
    expression = buildAddOp(buildOpaqueVarRefExp(SgName("ind_" + arg(i) + "_ptrs")), expression);
    expression = buildAssignOp(buildOpaqueVarRefExp(SgName("ind_" + arg(i) + "_ptr")), expression);
    appendStatement(buildExprStatement(expression), ifBody);
  }

  // Calculate argument pointers
  expression = buildOpaqueVarRefExp(SgName("cur_offset"));
  for(unsigned int i=0; i<pl->args.size(); i++)
  {
    if(pl->args[i]->usesIndirection())
    {
      SgExpression* ex = buildAddOp(buildOpaqueVarRefExp(SgName(arg(i) + "_ptrs")), expression);
      ex = buildAssignOp(buildOpaqueVarRefExp(SgName(arg(i) + "_ptr")), ex);
      appendStatement(buildExprStatement(ex), ifBody);
    }
    else if(!pl->args[i]->consideredAsConst())
    {
      SgExpression* ex = buildMultiplyOp(expression, buildIntVal(pl->args[i]->dim));
      ex = buildAddOp(buildOpaqueVarRefExp(SgName(arg(i) + "_d")), ex);
      ex = buildAssignOp(buildOpaqueVarRefExp(SgName(arg(i))), ex);
      appendStatement(buildExprStatement(ex), ifBody);
    }
  }

  // Set Shared Memory Pointers
  for(unsigned int i=0; i<pl->planContainer.size(); i++)
  {
    if(i==0)
    {
      SgVariableDeclaration* var_dec = buildVariableDeclaration(SgName("nbytes"), buildIntType(), buildAssignInitializer(buildIntVal(0)), ifBody);
      appendStatement(var_dec, ifBody);
    }
    else
    {
      // Example: nbytes += ROUND_UP(ind_arg0_size*sizeof(float)*2);
      expression = buildMultiplyOp(buildSizeOfOp(buildFloatType()), buildIntVal(pl->planContainer[i-1]->dim));
      expression = buildMultiplyOp(buildOpaqueVarRefExp(SgName("ind_" + arg(i-1) + "_size")), expression);
      SgExprListExp* expressions = buildExprListExp();
      expressions->append_expression(expression);
      expression = buildFunctionCallExp(SgName("ROUND_UP"), buildIntType(), expressions);
      expression = buildPlusAssignOp(buildOpaqueVarRefExp(SgName("nbytes")), expression);
      appendStatement(buildExprStatement(expression), ifBody);
    }
    expression = buildOpaqueVarRefExp(SgName("shared[nbytes]"));
    expression = buildAddressOfOp(expression);
    expression = buildCastExp(expression, buildPointerType(pl->planContainer[i]->type));
    expression = buildAssignOp(buildOpaqueVarRefExp(SgName("ind_" + arg(i) + "_s")), expression);
    appendStatement(buildExprStatement(expression), ifBody);
  }
  appendStatement(threadCondition, kernelBody);

  // 4.2 CALL SYNCTHREADS
  // ========================================================
  SgExprStatement *kCall = buildFunctionCallStmt("__syncthreads", buildVoidType(), NULL, kernelBody);
  appendStatement(kCall, kernelBody);

  // 4.3 COPY INDIRECT DATA SETS INTO SHARED MEMORY
  // ========================================================
  for(unsigned int i=0; i<pl->planContainer.size(); i++)
  {
    // Create outer loop
    SgScopeStatement *loopBody = buildBasicBlock();
    SgStatement *loopInit = buildVariableDeclaration( SgName("n"), buildIntType(), buildAssignInitializer(buildOpaqueVarRefExp(SgName("threadIdx.x"))) );
    SgName loopVar = isSgVariableDeclaration(loopInit)->get_definition()->get_vardefn()->get_name();
    SgName loopVarLimit = SgName("ind_") + arg(i) + SgName("_size");
    SgExprStatement *loopTest = buildExprStatement( buildLessThanOp( buildVarRefExp(loopVar), buildOpaqueVarRefExp(loopVarLimit) ) );  
    SgPlusAssignOp *loopIncrement = buildPlusAssignOp(buildVarRefExp(loopVar), buildOpaqueVarRefExp(SgName("blockDim.x")) );
    SgStatement *loopForLoop = buildForStatement(loopInit, loopTest, loopIncrement, loopBody);
    
    // If dim is greater than one then we need to use cached variable
    if(  pl->planContainer[i]->dim > 1)
    {
      if(pl->planContainer[i]->access == OP_READ || pl->planContainer[i]->access == OP_RW)
      {
        // Create cached variable
        SgExpression* e = buildOpaqueVarRefExp(SgName("ind_") + arg(i) + SgName("_ptr[n]"));
        SgStatement *vdec = buildVariableDeclaration( SgName("ind_index"), buildIntType(), buildAssignInitializer(e) );
        appendStatement(vdec, loopBody);
      }

      // Create inner loop body
      for(int j=0; j<pl->planContainer[i]->dim; j++)
      {
        SgName indrvar = SgName("ind_") + arg(i) + SgName("_s[" + buildStr(j) + "+n*" + buildStr(pl->planContainer[i]->dim) + "]");
        SgExpression* asgnExpr;
        SgName asgnName; 
        switch(pl->planContainer[i]->access)
        {
          case OP_READ:
          case OP_RW:
            asgnName = SgName("ind_") + arg(i) + SgName("[" + buildStr(j) + "+ind_index*" + buildStr(pl->planContainer[i]->dim) + "]");
            asgnExpr = buildOpaqueVarRefExp(asgnName);
            break;
          case OP_WRITE:
            break;
          case OP_INC:
            asgnExpr = buildIntVal(0);
            break;
          default:
            break;
        }
        expression = buildAssignOp( buildOpaqueVarRefExp(indrvar), asgnExpr );
        appendStatement(buildExprStatement(expression), loopBody);
      }
    }
    else
    {
      SgName indrvar = SgName("ind_") + arg(i) + SgName("_s[n*" + buildStr(pl->planContainer[i]->dim) + "]");
      SgExpression* asgnExpr;
      SgName asgnName; 
      switch(pl->planContainer[i]->access)
      {
        case OP_READ:
        case OP_RW:
          asgnName = SgName("ind_") + arg(i) + SgName("[ind_") + arg(i) + SgName("_ptr[n]*") + SgName(buildStr(pl->planContainer[i]->dim)) + SgName("]");
          asgnExpr = buildOpaqueVarRefExp(asgnName);
          break;
        case OP_WRITE:
          break;
        case OP_INC:
          asgnExpr = buildIntVal(0);
          break;
        default:
          break;
      }
      expression = buildAssignOp( buildOpaqueVarRefExp(indrvar), asgnExpr );
      appendStatement(buildExprStatement(expression), loopBody);
    }
    // Append outer loop
    appendStatement(loopForLoop, kernelBody);
  }

  // 4.4 CALL SYNCTHREADS
  // ========================================================
  kCall = buildFunctionCallStmt("__syncthreads", buildVoidType(), NULL, kernelBody);
  appendStatement(kCall, kernelBody);
    
  // 5. PRE-KERNEL HEADER
  // ========================================================
  preKernelGlobalDataHandling(fn, pl);

  // 6. CREATE OUTER MAIN LOOP BODY
  // ========================================================
  SgScopeStatement *mainLoopBody = buildBasicBlock();
  SgStatement *mainLoopInit = buildVariableDeclaration( SgName("n"), buildIntType(), buildAssignInitializer(buildOpaqueVarRefExp(SgName("threadIdx.x"))) );
  SgName mainLoopVar = isSgVariableDeclaration(mainLoopInit)->get_definition()->get_vardefn()->get_name();

  // Part 1: Inside the main outer loop body - the first part, defining col2
  varDec = buildVariableDeclaration(SgName("col2"), buildIntType(), buildAssignInitializer(buildIntVal(-1)), mainLoopBody);
  SgName color2 = isSgVariableDeclaration(varDec)->get_definition()->get_vardefn()->get_name();
  appendStatement(varDec, mainLoopBody);

  // Part 2 <begin>: Create the if statement and do the actual calculation
  SgScopeStatement *condBody2 = buildBasicBlock();        
  SgExprStatement *condStmt2 = buildExprStatement( buildLessThanOp( buildVarRefExp(mainLoopVar), buildVarRefExp(SgName("nelem")) ) );
  SgIfStmt* cond2 = buildIfStmt(condStmt2, condBody2, NULL);
  
  // Part 2_1: Initialize Local Variables
  for(int i=0; i<pl->numArgs(); i++)
  {
    if(pl->args[i]->isNotGlobal() && pl->args[i]->access == OP_INC && pl->args[i]->usesIndirection())
    {
      for(int j=0; j<pl->args[i]->dim; j++)
      {
        // If uses indirection
        SgExpression* exprDst = buildOpaqueVarRefExp(argLocal(i) + SgName("["+buildStr(j)+"]"));
        SgExpression* exprSrc = buildIntVal(0);
      
        // Append statement to the inner loop body
        appendStatement(buildExprStatement( buildAssignOp( exprDst,exprSrc ) ), condBody2);
      }
    }
  }

  // Part 2_1_2: Load directly accessed global memory data into local variables i.e. registers
  for(int i=0; i<pl->numArgs(); i++)
  {
    if(pl->args[i]->isNotGlobal() && !pl->args[i]->usesIndirection())
    {
      if(pl->args[i]->access == OP_READ || pl->args[i]->access == OP_RW)
      {
        for(int j=0; j<pl->args[i]->dim; j++)
        {
          // (Example? old note? to be deleted?) arg4_l[0] = *(arg4 + n * 4 + 0);
        
          SgExpression* lhs1 = buildMultiplyOp( buildOpaqueVarRefExp(SgName("n")), buildIntVal(pl->args[i]->dim) );
          lhs1 = buildAddOp(lhs1, buildIntVal(j));
          lhs1 = buildAddOp(buildOpaqueVarRefExp(arg(i)), lhs1);
          lhs1 = buildPointerDerefExp(lhs1);

          SgExpression* rhs1 = buildOpaqueVarRefExp(argLocal(i) + SgName("[" + buildStr(j) + "]"));
          rhs1 = buildAssignOp(rhs1, lhs1);      

          SgStatement* expr_statement = buildExprStatement(rhs1);
          appendStatement(expr_statement, condBody2);
        }
      }
    }
  } 
  
  // Part 2_2: Call user kernel <!!COMPLICATED!!>
  SgExprListExp *kPars = buildExprListExp();
  for(int i=0; i<pl->numArgs(); i++)
  {
    if(pl->args[i]->isGlobal())
    {
      if(pl->args[i]->consideredAsReduction())
      {
        expression = buildOpaqueVarRefExp(SgName(arg(i) + "_l"));
        kPars->append_expression(expression);
      }
      else if(pl->args[i]->consideredAsConst())
      {
        expression = buildOpaqueVarRefExp(SgName(arg(i)));
        kPars->append_expression(expression);
      }
    }
    else if(pl->args[i]->isNotGlobal())
    {
      if(pl->args[i]->usesIndirection())
      {
        if(pl->args[i]->access == OP_INC)
        {
          expression = buildOpaqueVarRefExp(SgName(arg(i) + "_l"));
          kPars->append_expression(expression);
        }
        else
        {
          expression = buildMultiplyOp(buildOpaqueVarRefExp(SgName(arg(i)+"_ptr[n]")), buildIntVal(pl->args[i]->dim));
          expression = buildAddOp(buildOpaqueVarRefExp(SgName("ind_" + arg(pl->args[i]->plan_index) + "_s")), expression);
          kPars->append_expression(expression);
        }
      }
      else
      {
        kPars->append_expression( buildOpaqueVarRefExp(SgName(argLocal(i))) );
      }
    }
  }
  SgExprStatement *uf = buildFunctionCallStmt(SgName(kernel_name), buildVoidType(), kPars);
  appendStatement(uf, condBody2);

  // Part 2_2_2: Move directly accessed data back to registers
  for(int i=0; i<pl->numArgs(); i++)
  {
    if(pl->args[i]->isNotGlobal() && !pl->args[i]->usesIndirection())
    {
      if(pl->args[i]->access == OP_WRITE || pl->args[i]->access == OP_RW)
      {
        for(int j=0; j<pl->args[i]->dim; j++)
        {
          // (Example? old note? to be deleted?) arg4_l[0] = *(arg4 + n * 4 + 0);
        
          SgExpression* lhs1 = buildMultiplyOp( buildOpaqueVarRefExp(SgName("n")), buildIntVal(pl->args[i]->dim) );
          lhs1 = buildAddOp(lhs1, buildIntVal(j));
          lhs1 = buildAddOp(buildOpaqueVarRefExp(arg(i)), lhs1);
          lhs1 = buildPointerDerefExp(lhs1);

          SgExpression* rhs1 = buildOpaqueVarRefExp(argLocal(i) + SgName("[" + buildStr(j) + "]"));
          lhs1 = buildAssignOp(lhs1, rhs1);      

          SgStatement* expr_statement = buildExprStatement(lhs1);
          appendStatement(expr_statement, condBody2);
        }
      }
    }
  }
  
  // Part 2_3: Set the color of the thread
  expression = buildAssignOp( buildOpaqueVarRefExp(color2), buildOpaqueVarRefExp(SgName("color[") +  mainLoopVar + SgName("]")) );
  SgStatement* expr_statement = buildExprStatement(expression);
  appendStatement(expr_statement, condBody2);

  // Part 2 <end>: Add the condition body to the main loop body
  appendStatement(cond2, mainLoopBody);
  
  
  // 7. COPY DATA BACK TO THE SHARED MEMORY
  // ========================================================
  
  // Part 3: Inside the main outer loop body - the third part, copying values from arg to shared memory
  bool brequired = false;
  for(int i=0; i<pl->numArgs(); i++) {
      if(pl->args[i]->isNotGlobal() && pl->args[i]->access == OP_INC && pl->args[i]->usesIndirection()) {
        brequired = true;
      }
  }
  if(brequired)
  {
    // Create outer loop
    SgScopeStatement *loopBody = buildBasicBlock();
    SgStatement *loopInit = buildVariableDeclaration( SgName("col"), buildIntType(), buildAssignInitializer(buildIntVal(0)) );
    SgName loopVar = isSgVariableDeclaration(loopInit)->get_definition()->get_vardefn()->get_name();
    SgName loopVarLimit = SgName("ncolor");
    SgExprStatement *loopTest = buildExprStatement( buildLessThanOp( buildVarRefExp(loopVar), buildOpaqueVarRefExp(loopVarLimit) ) );  
    SgPlusPlusOp *loopIncrement = buildPlusPlusOp(buildVarRefExp(loopVar));
    SgStatement *loopForLoop = buildForStatement(loopInit, loopTest, loopIncrement, loopBody);

    // Create if color match condition
    SgScopeStatement *condBody = buildBasicBlock();        
    SgExprStatement *condStmt = buildExprStatement( buildEqualityOp( buildVarRefExp(loopVar), buildVarRefExp(color2) ) );
    SgIfStmt* cond = buildIfStmt(condStmt, condBody, NULL);

    bool alreadyDefined = false;
    for(int i=0; i<pl->numArgs(); i++)
    {
      if(pl->args[i]->isNotGlobal() && pl->args[i]->access == OP_INC && pl->args[i]->usesIndirection())
      {
        if(pl->args[i]->dim > 1)
        {
          // Create cached variable
          SgExpression* e = buildOpaqueVarRefExp(arg(i)+ SgName("_ptr[n]"));
          if(!alreadyDefined)
          {
            SgStatement *vdec = buildVariableDeclaration( SgName("ind_index"), buildIntType(), buildAssignInitializer(e) );
            appendStatement(vdec, condBody);
            alreadyDefined = true;
          }
          else
          {
            SgExpression* ee = buildAssignOp(buildOpaqueVarRefExp(SgName("ind_index")), e);
            appendStatement(buildExprStatement(ee), condBody);
          }
      
          // Create inner loop
          for(int j=0; j<pl->args[i]->dim; j++)
          {
            SgName dstName = SgName("ind_")+ arg(pl->args[i]->plan_index)+ SgName("_s[" + buildStr(j) + "+ind_index*" + buildStr(pl->args[i]->dim) + "]");
            SgName srcName = argLocal(i) + SgName("[" + buildStr(j) + "]");
            expression = buildPlusAssignOp( buildOpaqueVarRefExp(dstName), buildOpaqueVarRefExp(srcName) );
            appendStatement(buildExprStatement(expression), condBody);      
          }
        }
        else
        {
            SgName dstName = SgName("ind_")+ arg(pl->args[i]->plan_index)+ SgName("_s[")+ arg(i)+ SgName("_ptr[n]*" + buildStr(pl->args[i]->dim) + "]");
            SgName srcName = argLocal(i) + SgName("[0]");
            expression = buildPlusAssignOp( buildOpaqueVarRefExp(dstName), buildOpaqueVarRefExp(srcName) );
            appendStatement(buildExprStatement(expression), condBody);      
        }
      }
    }
    // Append condition statement to outer loop body
    appendStatement(cond, loopBody);
    // Create syncfunction statement
    kCall = buildFunctionCallStmt("__syncthreads", buildVoidType(), NULL, kernelBody);
    // Append syncthreads
    appendStatement(kCall, loopBody);
    // Append outer loop to the main body
    appendStatement(loopForLoop, mainLoopBody);
  }
  
  // Append main outer loop statement
  SgExprStatement *mainLoopTest = buildExprStatement( buildLessThanOp( buildVarRefExp(mainLoopVar), buildOpaqueVarRefExp(SgName("nelem2")) ) );  
  SgPlusAssignOp *mainLoopIncrement = buildPlusAssignOp(buildVarRefExp(mainLoopVar), buildOpaqueVarRefExp(SgName("blockDim.x")) );
  SgStatement *mainForLoop = buildForStatement(mainLoopInit, mainLoopTest, mainLoopIncrement, mainLoopBody);
  appendStatement(mainForLoop, kernelBody);
  
  

  // 8. COPY DATA BACK TO DRAM
  // ========================================================

  // For write and icrement
  // Copy indirect datasets into shared memory or zero increment
  for(unsigned int i=0; i<pl->planContainer.size(); i++)
  {
      if(pl->planContainer[i]->access == OP_READ)
        continue;
      if(pl->planContainer[i]->access == OP_MAX)
        continue;
      if(pl->planContainer[i]->access == OP_MIN)
        continue;
      
      // Create outer loop
      SgScopeStatement *loopBody = buildBasicBlock();
      SgStatement *loopInit = buildVariableDeclaration( SgName("n"), buildIntType(), buildAssignInitializer(buildOpaqueVarRefExp(SgName("threadIdx.x"))) );
      SgName loopVar = isSgVariableDeclaration(loopInit)->get_definition()->get_vardefn()->get_name();
      SgName loopVarLimit = SgName("ind_") + arg(i) + SgName("_size");
      SgExprStatement *loopTest = buildExprStatement( buildLessThanOp( buildVarRefExp(loopVar), buildOpaqueVarRefExp(loopVarLimit) ) );  
      SgPlusAssignOp *loopIncrement = buildPlusAssignOp(buildOpaqueVarRefExp(loopVar), buildOpaqueVarRefExp(SgName("blockDim.x")) );
      SgStatement *loopForLoop = buildForStatement(loopInit, loopTest, loopIncrement, loopBody);

      if(pl->planContainer[i]->dim > 1)
      {
        // Create cached variable
        SgExpression* e = buildOpaqueVarRefExp(SgName("ind_") + arg(i) + SgName("_ptr[n]"));
        SgStatement *vdec = buildVariableDeclaration( SgName("ind_index"), buildIntType(), buildAssignInitializer(e) );
        appendStatement(vdec, loopBody);
        
        for(int j=0; j<pl->planContainer[i]->dim; j++)
        {
          SgExpression* rhs = buildOpaqueVarRefExp(SgName("ind_" + arg(i)) + SgName("_s[" + buildStr(j) + "+n*" + buildStr(pl->planContainer[i]->dim) + "]"));
          SgExpression* lhs = buildOpaqueVarRefExp(SgName("ind_" + arg(i)) + SgName("[" + buildStr(j) + "+ind_index*" + buildStr(pl->planContainer[i]->dim) + "]"));
          expression = buildAssignOp( lhs, rhs );
          switch(pl->planContainer[i]->access)
          {
          case OP_RW:
          case OP_WRITE:
            expression = buildAssignOp( lhs, rhs );  
            break;
          case OP_INC:
            expression = buildPlusAssignOp( lhs, rhs );
            break;
          default:
            break;
          }
          appendStatement(buildExprStatement(expression), loopBody);
        }
      }
      else
      {
        SgExpression* rhs = buildOpaqueVarRefExp(SgName("ind_" + arg(i)) + SgName("_s[n*" + buildStr(pl->planContainer[i]->dim) + "]"));
        SgExpression* lhs = buildOpaqueVarRefExp(SgName("ind_" + arg(i)) + SgName("[ind_" + arg(i) + "_ptr[n]*" + buildStr(pl->planContainer[i]->dim) + "]"));
        expression = buildAssignOp( lhs, rhs );
        appendStatement(buildExprStatement(expression), loopBody);
      }
      // Append outer loop to the main body
      appendStatement(loopForLoop, kernelBody);
  }

  // Handle post global data handling
  postKernelGlobalDataHandling(fn, pl);

  generateStandardStub(fn, kernel_name, pl);
}

void OPParLoop::generateStandardStub(SgFunctionCallExp *fn, string kernel_name, op_par_loop_args *pl)
{
  // The following code builds the stub function.

  // As usual we build a list of parameters for the function.
  SgFunctionParameterList *paramList = buildFunctionParameterList();
  SgInitializedName *name = buildInitializedName(SgName("name"), buildPointerType(buildConstType(buildCharType())));
  appendArg(paramList, name);
  name = buildInitializedName(SgName("set"), op_set);
  appendArg(paramList, name);
  // Add other arguments
  for(int i=0; i<pl->numArgs(); i++)
  {
    name = buildInitializedName(SgName("arg"+buildStr(i)), buildPointerType(op_dat));
    appendArg(paramList, name);
    name = buildInitializedName(SgName("idx"+buildStr(i)), buildIntType());
    appendArg(paramList, name);
    name = buildInitializedName(SgName("ptr"+buildStr(i)), buildPointerType(op_ptr));
    appendArg(paramList, name);
    name = buildInitializedName(SgName("acc"+buildStr(i)), op_access);
    appendArg(paramList, name);
  }

  createStub(kernel_name, paramList);

  // Add variables nargs and 'ninds'
  //Example : int nargs = 3, ninds = 2;
  
  SgVariableDeclaration *varDec = buildVariableDeclaration(SgName("nargs"), buildIntType(), buildAssignInitializer(buildIntVal(pl->numArgs())), stubBody);
  appendStatement(varDec,stubBody);
  varDec = buildVariableDeclaration(SgName("ninds"), buildIntType(), buildAssignInitializer(buildIntVal(pl->planContainer.size())), stubBody);
  appendStatement(varDec,stubBody);

  // Add maximum grid size
  // Example : int gridsize = (set.size - 1) / BSIZE + 1;
  SgExpression *expresn = buildOpaqueVarRefExp(SgName("set.size"));
  expresn = buildSubtractOp(expresn, buildIntVal(1));
  expresn = buildDivideOp(expresn, buildOpaqueVarRefExp(SgName("BSIZE")));
  expresn = buildAddOp(expresn, buildIntVal(1));
  varDec = buildVariableDeclaration(SgName("gridsize"), buildIntType(), buildAssignInitializer(expresn));
  appendStatement(varDec, stubBody);
  
  // Add plan variables
  SgExprListExp* exprList_args = buildExprListExp();
  SgExprListExp* exprList_idxs = buildExprListExp();
  SgExprListExp* exprList_ptrs = buildExprListExp();
  SgExprListExp* exprList_dims = buildExprListExp();
  SgExprListExp* exprList_accs = buildExprListExp();
  SgExprListExp* exprList_inds = buildExprListExp();

  // Use the counter to indentify the index, e.g. 1st argument to use indireciton, or 2nd argument to use indirection
  // and we keep incrementing the counter. So the first arg to use indirection will get 0 and second arg to use indirection
  // will get 1.
  for(int i = 0; i < pl->numArgs(); i++)
  {
    exprList_args->append_expression( buildPointerDerefExp(buildOpaqueVarRefExp(SgName("arg"+buildStr(i)))) );
    if(pl->args[i]->usesIndirection())
    {
      exprList_idxs->append_expression( buildOpaqueVarRefExp(SgName("idx"+buildStr(i))) );
      exprList_ptrs->append_expression( buildPointerDerefExp(buildOpaqueVarRefExp(SgName("ptr"+buildStr(i)))) );
    }
    else
    {
      exprList_idxs->append_expression( buildIntVal(-1) );
      exprList_ptrs->append_expression( buildOpaqueVarRefExp(SgName("OP_ID")) );
    }
    exprList_dims->append_expression( buildOpaqueVarRefExp(SgName(arg(i)+"->dim"))     );
    exprList_accs->append_expression( buildOpaqueVarRefExp(SgName("acc"+buildStr(i))) );
    exprList_inds->append_expression( buildIntVal(pl->args[i]->plan_index) );
  }
  varDec = buildVariableDeclaration( SgName("args"), buildArrayType(op_dat, buildIntVal(pl->numArgs())), buildAggregateInitializer(exprList_args), stubBody);
  appendStatement(varDec,stubBody);
  varDec = buildVariableDeclaration( SgName("idxs"), buildArrayType(buildIntType(), buildIntVal(pl->numArgs())), buildAggregateInitializer(exprList_idxs), stubBody);
  appendStatement(varDec,stubBody);
  varDec = buildVariableDeclaration( SgName("ptrs"), buildArrayType(op_ptr, buildIntVal(pl->numArgs())), buildAggregateInitializer(exprList_ptrs), stubBody);
  appendStatement(varDec,stubBody);
  varDec = buildVariableDeclaration( SgName("dims"), buildArrayType(buildIntType(), buildIntVal(pl->numArgs())), buildAggregateInitializer(exprList_dims), stubBody);
  appendStatement(varDec,stubBody);
  varDec = buildVariableDeclaration( SgName("accs"), buildArrayType(op_access, buildIntVal(pl->numArgs())), buildAggregateInitializer(exprList_accs), stubBody);
  appendStatement(varDec,stubBody);
  varDec = buildVariableDeclaration( SgName("inds"), buildArrayType(buildIntType(), buildIntVal(pl->numArgs())), buildAggregateInitializer(exprList_inds), stubBody);
  appendStatement(varDec,stubBody);
  

  // Create and initialize the Plan variable pointer
  //Example: op_plan *Plan = plan(name,set,nargs,args,idxs,ptrs,dims,typs,accs,ninds,inds);
  // Create the plan function call, 1) first create params, 2) then call the function
  SgExprListExp *planPars = buildExprListExp();
  planPars->append_expression(buildOpaqueVarRefExp(SgName("name")));
  planPars->append_expression(buildOpaqueVarRefExp(SgName("set")));
  planPars->append_expression(buildOpaqueVarRefExp(SgName("nargs")));
  planPars->append_expression(buildOpaqueVarRefExp(SgName("args")));
  planPars->append_expression(buildOpaqueVarRefExp(SgName("idxs")));
  planPars->append_expression(buildOpaqueVarRefExp(SgName("ptrs")));
  planPars->append_expression(buildOpaqueVarRefExp(SgName("dims")));
  planPars->append_expression(buildOpaqueVarRefExp(SgName("accs")));
  planPars->append_expression(buildOpaqueVarRefExp(SgName("ninds")));
  planPars->append_expression(buildOpaqueVarRefExp(SgName("inds")));
  SgFunctionCallExp *expPlanFunc = buildFunctionCallExp(SgName("plan"), op_plan, planPars);
  
  // 3) then as the initializer of the Plan variable
  varDec = buildVariableDeclaration( SgName("Plan"), buildPointerType(op_plan),  buildAssignInitializer(expPlanFunc), stubBody);
  appendStatement(varDec,stubBody);

  // Add block offset
  varDec = buildVariableDeclaration(SgName("block_offset"), buildIntType(), buildAssignInitializer(buildIntVal(0)), stubBody);
  appendStatement(varDec,stubBody);

  preHandleConstAndGlobalData(fn, pl);

  // Add Total Time
  varDec = buildVariableDeclaration(SgName("total_time"), buildFloatType(), buildAssignInitializer(buildFloatVal(0.0f)), stubBody);
  appendStatement(varDec,stubBody);

  // Add for loop for executing op_cuda_res<<<gridsize,bsize,nshared>>>
  // Create loop body
  SgScopeStatement *blockLoopBody = buildBasicBlock();
  SgStatement *blockLoopInit = buildVariableDeclaration(SgName("col"), buildIntType(), buildAssignInitializer(buildIntVal(0)));
  SgName blockLoopVar = isSgVariableDeclaration(blockLoopInit)->get_definition()->get_vardefn()->get_name();
  
  // Add nshared and nblocks
  SgExpression* e = buildOpaqueVarRefExp(SgName("Plan->ncolblk[") + blockLoopVar + SgName("]"));
  varDec = buildVariableDeclaration(SgName("nblocks"), buildIntType(), buildAssignInitializer(e), blockLoopBody);
  appendStatement(varDec,blockLoopBody);
  e = buildOpaqueVarRefExp(SgName("Plan->nshared"));
  varDec = buildVariableDeclaration(SgName("nshared"), buildIntType(), buildAssignInitializer(e), blockLoopBody);
  appendStatement(varDec,blockLoopBody);

  createBeginTimerBlock(blockLoopBody);
  
  // To add a call to the CUDA function, we need to build a list of parameters that
  // we pass to it. The easiest way to do this is to name the members of the 
  // struct to which they belong, but this is not the most elegant approach.
  SgExprListExp *kPars = buildExprListExp();
  {
    // First Assemble all expressions using plan container <for arguments with indirection>
    for(unsigned int i=0; i<pl->planContainer.size(); i++)
    {
      SgExpression *e = buildOpaqueVarRefExp(SgName("arg"+buildStr(pl->planContainer[i]->own_index)+"->dat_d"));
      e = buildCastExp(e, buildPointerType(pl->planContainer[i]->type));
      kPars->append_expression(e);
      e = buildOpaqueVarRefExp(SgName("Plan->ind_ptrs["+buildStr(i)+"]"));
      kPars->append_expression(e);
      e = buildOpaqueVarRefExp(SgName("Plan->ind_sizes["+buildStr(i)+"]"));
      kPars->append_expression(e);
      e = buildOpaqueVarRefExp(SgName("Plan->ind_offs["+buildStr(i)+"]"));
      kPars->append_expression(e);
    }
    // Then add all the pointers
    for(unsigned int i=0; i<pl->args.size(); i++)
    {
      if(pl->args[i]->usesIndirection())
      {
        SgExpression *e = buildOpaqueVarRefExp(SgName("Plan->ptrs["+buildStr(i)+"]"));
        kPars->append_expression(e);
      }
      else
      {
        SgExpression *e = buildOpaqueVarRefExp(SgName("arg"+buildStr(i)+"->dat_d"));
        e = buildCastExp(e, buildPointerType(pl->args[i]->type));
        kPars->append_expression(e);
      }
    }
    // Add additional parameters
    e = buildOpaqueVarRefExp(SgName("block_offset"));
    kPars->append_expression(e);
    e = buildOpaqueVarRefExp(SgName("Plan->blkmap"));
    kPars->append_expression(e);
    e = buildOpaqueVarRefExp(SgName("Plan->offset"));
    kPars->append_expression(e);
    e = buildOpaqueVarRefExp(SgName("Plan->nelems"));
    kPars->append_expression(e);
    e = buildOpaqueVarRefExp(SgName("Plan->nthrcol"));
    kPars->append_expression(e);
    e = buildOpaqueVarRefExp(SgName("Plan->thrcol"));
    kPars->append_expression(e);
    if(reduction_required)
      kPars->append_expression(buildOpaqueVarRefExp(SgName("block_reduct")));
  }

  // We have to add the kernel configuration as part of the function name
  // as CUDA is not directly supported by ROSE - however, I understand
  // that CUDA and OpenCL support is coming soon!
  SgExprStatement *kCall = buildFunctionCallStmt("op_cuda_"+kernel_name+"<<<nblocks,BSIZE,nshared>>>", buildVoidType(), kPars, stubBody);
  appendStatement(kCall,blockLoopBody);

  createEndTimerBlock(blockLoopBody, true);

  // Call cuda thread synchronize
  kCall = buildFunctionCallStmt("cudaThreadSynchronize", buildVoidType(), NULL, stubBody);
  appendStatement(kCall, blockLoopBody);
 
  // Increment the block_offset now
  e = buildPlusAssignOp( buildOpaqueVarRefExp(SgName("block_offset")), buildOpaqueVarRefExp(SgName("nblocks")) );
  appendStatement(buildExprStatement(e),blockLoopBody);
  
  // We can build a test and an increment for the loop, then insert 
  // the loop into the body of the outer loop.
  SgExprStatement *blockLoopTest = buildExprStatement( buildLessThanOp( buildVarRefExp(blockLoopVar), buildOpaqueVarRefExp(SgName("Plan->ncolors")) ) );
  SgPlusPlusOp *blockLoopIncrement = buildPlusPlusOp(buildVarRefExp(blockLoopVar));
  SgStatement *blockLoopForLoop = buildForStatement(blockLoopInit, blockLoopTest, blockLoopIncrement, blockLoopBody);
  appendStatement(blockLoopForLoop,stubBody);

  postHandleConstAndGlobalData(fn, pl);

  // Add return statement
  SgReturnStmt* rtstmt = buildReturnStmt(buildOpaqueVarRefExp(SgName("total_time")));
  appendStatement(rtstmt, stubBody);
}



void OPParLoop::preKernelGlobalDataHandling(SgFunctionCallExp *fn, op_par_loop_args *pl)
{
  // HANDLE GLOBAL DATA <TRANSFER TO DEVICE>
  for(int i=0; i<pl->numArgs(); i++)
  {
    SgStatement *viInit = buildVariableDeclaration(SgName("d"), buildIntType(), buildAssignInitializer(buildIntVal(0)));
    SgName indVar = isSgVariableDeclaration(viInit)->get_definition()->get_vardefn()->get_name();

    if(pl->args[i]->consideredAsReduction())
    {
      // Build the body of the loop.
      SgExpression *lhs, *rhs;
      lhs = buildPntrArrRefExp(buildVarRefExp(argLocal(i), kernelBody), buildVarRefExp(indVar));
      switch(pl->args[i]->access)
      {
        case OP_INC:
          rhs = buildIntVal(0);
          break;
        default:
          rhs = buildPntrArrRefExp(buildVarRefExp(arg(i), kernelBody), buildVarRefExp(indVar));
          break;
      }
      SgStatement *action = buildAssignStatement(lhs, rhs);
      SgStatement *viLoopBody = buildBasicBlock(action);

      // We can build a test and an increment for the loop, then insert 
      // the loop into the body of the outer loop.
      SgExprStatement *viTest = buildExprStatement(buildLessThanOp(buildVarRefExp(indVar), buildIntVal(pl->args[i]->dim)));
      SgPlusPlusOp *viIncrement = buildPlusPlusOp(buildVarRefExp(indVar));
      SgStatement *viForLoop = buildForStatement(viInit, viTest, viIncrement, viLoopBody);
      appendStatement(viForLoop,kernelBody);
    }
  }
}

void OPParLoop::postKernelGlobalDataHandling(SgFunctionCallExp *fn, op_par_loop_args *pl)
{
  // 4 HANDLE GLOBAL DATA <TRANSFER FROM DEVICE>
  for(int i=0; i<pl->numArgs(); i++)
  {
    if(pl->args[i]->consideredAsReduction())
    {
      // Create loop
      SgStatement *viInit = buildVariableDeclaration(SgName("d"), buildIntType(), buildAssignInitializer(buildIntVal(0)));
      SgName indVar = isSgVariableDeclaration(viInit)->get_definition()->get_vardefn()->get_name();
            
      // Call function
      SgExprListExp *kPars1 = buildExprListExp();
      SgExpression *e = buildOpaqueVarRefExp(arg(i));
      e = buildAddOp(e, buildVarRefExp(indVar));
      kPars1->append_expression(e);
      e = buildOpaqueVarRefExp(argLocal(i) + SgName("[d]"));
      kPars1->append_expression(e);
      e = buildOpaqueVarRefExp(SgName("block_reduct" + buildStr(i)));
      kPars1->append_expression(e);

      SgExprStatement *uf1 = NULL;
      switch(pl->args[i]->access)
      {
        case OP_INC:
          uf1 = buildFunctionCallStmt(SgName("op_reduction2_1<OP_INC>"), buildVoidType(), kPars1);
          break;
        case OP_MAX:
          uf1 = buildFunctionCallStmt(SgName("op_reduction2_1<OP_MAX>"), buildVoidType(), kPars1);
          break;
        case OP_MIN:
          uf1 = buildFunctionCallStmt(SgName("op_reduction2_1<OP_MIN>"), buildVoidType(), kPars1);
          break;
        default:
          break;
      }

      // build test and increment, and add the loop into the body of the inner loop.
      SgScopeStatement *viLoopBody = buildBasicBlock(uf1);
      SgExprStatement *viTest = buildExprStatement(buildLessThanOp(buildOpaqueVarRefExp(indVar), buildIntVal(pl->args[i]->dim)));
      SgPlusPlusOp *viIncrement = buildPlusPlusOp(buildOpaqueVarRefExp(indVar));
      SgStatement *viForLoop = buildForStatement(viInit, viTest, viIncrement, viLoopBody);
      appendStatement(viForLoop,kernelBody);
    }
  }
}




void OPParLoop::preHandleConstAndGlobalData(SgFunctionCallExp *fn, op_par_loop_args *pl)
{
  // Handle Reduct
  ///////////////////////

  bool required = false;
  SgVariableDeclaration* varDec = buildVariableDeclaration(SgName("reduct_bytes"), buildIntType(), buildAssignInitializer(buildIntVal(0)), stubBody);
  SgName varName = isSgVariableDeclaration(varDec)->get_definition()->get_vardefn()->get_name();  
  appendStatement(varDec,stubBody);

  SgVariableDeclaration* varDec2 = buildVariableDeclaration(SgName("reduct_size"), buildIntType(), buildAssignInitializer(buildIntVal(0)), stubBody);
  SgName varName2 = isSgVariableDeclaration(varDec2)->get_definition()->get_vardefn()->get_name();  
  appendStatement(varDec2,stubBody);

  SgExpression* varExp = buildVarRefExp(varName, stubBody);
  SgExpression* varExp2 = buildVarRefExp(varName2, stubBody);
  for(unsigned int i = 0; i < pl->args.size(); i++)
  {
    if(pl->args[i]->consideredAsReduction())
    {
      required = true;
      SgExpression* rhs =  buildMultiplyOp(buildIntVal(pl->args[i]->dim), buildSizeOfOp(pl->args[i]->type));
      SgExprListExp* list = buildExprListExp();
      list->append_expression(rhs);
      rhs = buildFunctionCallExp(SgName("ROUND_UP"), buildIntType(), list);
      SgExpression* expr = buildPlusAssignOp(varExp , rhs);
      appendStatement(buildExprStatement(expr), stubBody);

      list = buildExprListExp();
      list->append_expression(varExp2);
      list->append_expression(buildSizeOfOp(pl->args[i]->type));
      rhs = buildFunctionCallExp(SgName("MAX"), buildIntType(), list);
      expr = buildAssignOp(varExp2, rhs);
      appendStatement(buildExprStatement(expr), stubBody);
    }
  }

  SgExpression* expShared = buildMultiplyOp( varExp2, buildDivideOp(buildOpaqueVarRefExp(SgName("BSIZE")), buildIntVal(2)) );
  SgVariableDeclaration* varDec3 = buildVariableDeclaration(SgName("reduct_shared"), buildIntType(), buildAssignInitializer(expShared), stubBody);
  SgName varName3 = isSgVariableDeclaration(varDec2)->get_definition()->get_vardefn()->get_name();  
  appendStatement(varDec3,stubBody);

  if(required)
  {
    // call reallocReductArrays(reduct_bytes);
    SgExprListExp* kPars = buildExprListExp();
    kPars->append_expression(varExp);
    SgStatement *kCall = buildFunctionCallStmt("reallocReductArrays", buildVoidType(), kPars, stubBody);
      appendStatement(kCall, stubBody);

    // fixup with reduct_bytes
    SgExpression* expr = buildAssignOp(varExp, buildIntVal(0));
    appendStatement(buildExprStatement(expr), stubBody);
    
    for(unsigned int i = 0; i < pl->args.size(); i++)
    {
      if(pl->args[i]->consideredAsReduction())
      {
        kPars = buildExprListExp();
        kPars->append_expression((buildOpaqueVarRefExp(SgName("*"+arg(i)))));
        kPars->append_expression(varExp);
        kCall = buildFunctionCallStmt("push_op_dat_as_reduct", buildVoidType(), kPars, stubBody);
        appendStatement(kCall,stubBody);

        expr =  buildMultiplyOp(buildIntVal(pl->args[i]->dim), buildSizeOfOp(pl->args[i]->type));
        SgExprListExp* expressions = buildExprListExp();
        expressions->append_expression(expr);
        expr = buildFunctionCallExp(SgName("ROUND_UP"), buildIntType(), expressions);
        expr = buildPlusAssignOp(varExp , expr);
        appendStatement(buildExprStatement(expr), stubBody);
      }
    }

    // call mvReductArraysToDevice(reduct_bytes)
    kPars = buildExprListExp();
    kPars->append_expression(varExp);
    kCall = buildFunctionCallStmt("mvReductArraysToDevice", buildVoidType(), kPars, stubBody);
    appendStatement(kCall,stubBody);

    // handling global reduction - requires global memory allocation
    for(unsigned int i = 0; i < pl->args.size(); i++)
    {
      if(pl->args[i]->consideredAsReduction())
      {
        // declare block_reduct
        varDec = buildVariableDeclaration(SgName("block_reduct" + buildStr(i)), buildPointerType(buildVoidType()), buildAssignInitializer(buildIntVal(0)));
        appendStatement(varDec, stubBody);

        // allocate memory
        kPars = buildExprListExp();
        kPars->append_expression(buildAddressOfOp( buildOpaqueVarRefExp(SgName("block_reduct") + buildStr(i)) ));
        kPars->append_expression(buildMultiplyOp( buildOpaqueVarRefExp(SgName("gridsize")), buildSizeOfOp(pl->args[i]->type) ));
        kCall = buildFunctionCallStmt("cudaMalloc", buildVoidType(), kPars, stubBody);
        appendStatement(kCall,stubBody);
      }
    }
  }


  // Handle Const
  ///////////////////////

  required = false;
  varDec = buildVariableDeclaration(SgName("const_bytes"), buildIntType(), buildAssignInitializer(buildIntVal(0)), stubBody);
  varName = isSgVariableDeclaration(varDec)->get_definition()->get_vardefn()->get_name();  
  appendStatement(varDec,stubBody);
  varExp = buildVarRefExp(varName, stubBody);
  for(unsigned int i = 0; i < pl->args.size(); i++)
  {
    if(pl->args[i]->consideredAsConst())
    {
      required = true;
      SgExpression* rhs =  buildMultiplyOp(buildIntVal(pl->args[i]->dim), buildSizeOfOp(pl->args[i]->type));
      SgExprListExp* expressions = buildExprListExp();
      expressions->append_expression(rhs);
      rhs = buildFunctionCallExp(SgName("ROUND_UP"), buildIntType(), expressions);
      SgExpression* expr = buildPlusAssignOp(varExp , rhs);
      appendStatement(buildExprStatement(expr), stubBody);
    }
  }

  if(required)
  {
    // call reallocConstArrays(reduct_bytes);
    SgExprListExp* kPars = buildExprListExp();
    kPars->append_expression(varExp);
    SgStatement *kCall = buildFunctionCallStmt("reallocConstArrays", buildVoidType(), kPars, stubBody);
    appendStatement(kCall, stubBody);

    // fixup with reduct_bytes
    SgExpression* expr = buildAssignOp(varExp, buildIntVal(0));
    appendStatement(buildExprStatement(expr), stubBody);
    
    for(unsigned int i = 0; i < pl->args.size(); i++)
    {
      if(pl->args[i]->consideredAsConst())
      {
        kPars = buildExprListExp();
        kPars->append_expression((buildOpaqueVarRefExp(SgName("*"+arg(i)))));
        kPars->append_expression(varExp);
        kCall = buildFunctionCallStmt("push_op_dat_as_const", buildVoidType(), kPars, stubBody);
        appendStatement(kCall,stubBody);

        expr =  buildMultiplyOp(buildIntVal(pl->args[i]->dim), buildSizeOfOp(pl->args[i]->type));
        SgExprListExp* expressions = buildExprListExp();
        expressions->append_expression(expr);
        expr = buildFunctionCallExp(SgName("ROUND_UP"), buildIntType(), expressions);
        expr = buildPlusAssignOp(varExp , expr);
        appendStatement(buildExprStatement(expr), stubBody);
      }
    }

    // call mvReductArraysToDevice(reduct_bytes)
    kPars = buildExprListExp();
    kPars->append_expression(varExp);
    kCall = buildFunctionCallStmt("mvConstArraysToDevice", buildVoidType(), kPars, stubBody);
    appendStatement(kCall,stubBody);
  }
}

void OPParLoop::postHandleConstAndGlobalData(SgFunctionCallExp *fn, op_par_loop_args *pl)
{
  // Handle Reduct
  ///////////////////////

  bool required = false;
  for(unsigned int i = 0; i < pl->args.size(); i++)
  {
    if(pl->args[i]->consideredAsReduction())
    {
      required = true;
      break;
    }
  }
  if(required)
  {
    // call reallocReductArrays(reduct_bytes);
    SgExprListExp* kPars = buildExprListExp();
    kPars->append_expression(buildOpaqueVarRefExp(SgName("reduct_bytes")));
    SgStatement *kCall = buildFunctionCallStmt("mvReductArraysToHost", buildVoidType(), kPars, stubBody);
    appendStatement(kCall, stubBody);

    for(unsigned int i = 0; i < pl->args.size(); i++)
    {
      if(pl->args[i]->consideredAsReduction())
      {
        kPars = buildExprListExp();
        kPars->append_expression(buildOpaqueVarRefExp(SgName("*"+arg(i))));
        kCall = buildFunctionCallStmt("pop_op_dat_as_reduct", buildVoidType(), kPars, stubBody);
         appendStatement(kCall, stubBody);

        // free block_reduct memory
        kPars = buildExprListExp();
        kPars->append_expression( buildOpaqueVarRefExp(SgName("block_reduct" + buildStr(i))));
        kCall = buildFunctionCallStmt("cudaFree", buildVoidType(), kPars, stubBody);
        appendStatement(kCall,stubBody);
      }
    }
  }

  // Handle Const
  ///////////////////////
  
  // We dont need to do anything here for const
}


