/*
 * 
 * Copyright (c) 2010, Graham Markall
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
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

#ifndef ROSE_OP2PARLOOP_H
#define ROSE_OP2PARLOOP_H

#include <rose.h>
#include <vector>
#include <map>
#include <string>

#include "rose_op2argument.h"

using namespace std;
using namespace SageBuilder;
using namespace SageInterface;

class op_par_loop_args
{
  public:
    static const int num_params;
    string prev_name;

    SgFunctionRefExp *kernel;
    SgVarRefExp *set;
    // All args
    vector<op_argument*> args;          
    // Args that do uses indirection
    vector<op_argument*> ind_args;      
    // Arguments are passed into the loop in such a way so that args that uses
    // same op_dat reside together.  The planContainer stores the starting argument
    // of a op_dat category. The size of the container will therefore give us
    // the number of category there are. This neglects all arguments that do not
    // use indirection.
    vector<op_argument*> planContainer;  

    void init(SgFunctionCallExp* fn);

    // These violate ODR. Fix later.
    int numParams() { return num_params; }
    int numArgs() { return args.size(); }
    int numIndArgs() { return ind_args.size(); }

    void updatePlanContainer(op_argument* arg);

};

// The ParLoop class inherits from AstSimpleProcessing, which provides a mechanism for 
// traversing the AST of the program in ROSE. We use this for convenience, as opposed to
// trying to walk through the AST "manually".

class OPParLoop : public AstSimpleProcessing 
{
  public:
    map<string, SgFunctionDeclaration*> cudaFunctionDeclarations;
    
    virtual void visit(SgNode *n);
    virtual void atTraversalEnd();
    
    void setProject(SgProject *p);
    void unparse();
    void generateGlobalKernelsHeader();
 
  private:
    SgProject *project;
    SgGlobal *fileGlobalScope;
    vector<SgProject*> kernels;

    SgType *op_set, *op_dat, *op_ptr, *op_access, *op_plan;
    SgBasicBlock *kernelBody, *stubBody, *reductionBody;
    
    bool reduction_required;

    void generateSpecial(SgFunctionCallExp *fn, op_par_loop_args *pl);
    void generateStandard(SgFunctionCallExp *fn, op_par_loop_args *pl);
    string getName(SgFunctionRefExp *fn);

    void preHandleConstAndGlobalData(SgFunctionCallExp *fn, op_par_loop_args *pl);
    void postHandleConstAndGlobalData(SgFunctionCallExp *fn, op_par_loop_args *pl);
    void preKernelGlobalDataHandling(SgFunctionCallExp *fn, op_par_loop_args *pl);  
    void postKernelGlobalDataHandling(SgFunctionCallExp *fn, op_par_loop_args *pl);

    void createKernelFile(string kernel_name);
    void initialiseDataTypes();
    void createKernel(string kernel_name, SgFunctionParameterList *paramList);
    void createStub(string kernel_name, SgFunctionParameterList *paramList);
    void createReductionKernel(string kernel_name, SgFunctionParameterList *paramList);
    void createSharedVariable(string name, SgType *type, SgScopeStatement *scope);
    void createBeginTimerBlock(SgScopeStatement *scope);
    void createEndTimerBlock(SgScopeStatement *scope, bool accumulateTime);
    SgFunctionParameterList* createSpecialParameters(op_par_loop_args *pl);
    SgFunctionParameterList* createStandardParameters(op_par_loop_args *pl);
    SgFunctionParameterList* createReductionParameters(op_par_loop_args *pl);
    SgFunctionParameterList* buildSpecialStubParameters(op_par_loop_args *pl);
    void createSharedVariableDeclarations(op_par_loop_args *pl);
    void generateSpecialStub(SgFunctionCallExp *fn, string kernel_name, op_par_loop_args *pl);
    void generateStandardStub(SgFunctionCallExp *fn, string kernel_name, op_par_loop_args *pl);
    void generateReductionKernel(string kernel_name, op_par_loop_args *pl);
    void createReductionKernelCall(string kernel_name, op_par_loop_args *pl);
    void createSpecialKernelCall(string kernel_name, op_par_loop_args *pl);
    void createSpecialStubVariables();
    void createSpecialKernelVariables(op_par_loop_args *pl);
    void createStandardKernelVariables(op_par_loop_args *pl);
    void createSpecialUserFunctionCall(string kernel_name, op_par_loop_args *pl, SgName& induction_variable, SgScopeStatement *scope);
    SgName createSpecialKernelLoopConstruct(string kernel_name, op_par_loop_args* pl, SgScopeStatement* loopBody);
    void createSpecialKernelExecutionLoop(string kernel_name, op_par_loop_args *pl);
    void createSharedVariableOffsetInitialiser(op_par_loop_args *pl);
    void createSyncthreadsCall();
    void createCopyToShared(op_par_loop_args *pl);
    void createCopyFromShared(op_par_loop_args *pl);
    void createInitialiseLocalVariables(op_par_loop_args *pl, SgScopeStatement *scope);
    void createLoadDataToLocalVariables(op_par_loop_args *pl, SgScopeStatement *scope);
    SgScopeStatement* createStandardKernelExecutionLoop(const SgName& mainLoopVar);
    SgScopeStatement* createLessThanNumElemConditional(const SgName& mainLoopVar, SgScopeStatement *scope);
};

#endif
